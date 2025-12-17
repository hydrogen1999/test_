#!/bin/bash
# Submit jobs from a prioritized list while respecting a SLURM job limit

set -o errexit
set -o nounset
set -o pipefail

# Hard-coded job limit (running + pending)
JOB_LIMIT=2

# Default job list file (can be overridden by first argument)
JOB_LIST_FILE=${1:-job_list2.txt}

echo "============================================"
echo " SUBMITTING ISING JOBS WITH LIMIT: $JOB_LIMIT"
echo " Job list: $JOB_LIST_FILE"
echo "============================================"

# Move to the directory that contains this script
cd "$(dirname "$0")"

# Validate prerequisites
if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch command not found in PATH." >&2
  exit 1
fi

if ! command -v squeue >/dev/null 2>&1; then
  echo "Error: squeue command not found in PATH." >&2
  exit 1
fi

if [[ ! -f "$JOB_LIST_FILE" ]]; then
  echo "Error: job list file not found: $JOB_LIST_FILE" >&2
  exit 1
fi

# Ensure directories
mkdir -p cluster_jobs/done
mkdir -p cluster_logs

AGG_OUT_FILE="cluster_logs/ALL_OUTPUTS2.out"
AGG_ERR_FILE="cluster_logs/ALL_ERRORS2.err"

print_eq_box() {
  local job_name="$1"
  local job_id="$2"
  local submitted="$3"
  local ended="$4"
  local line1="Job name: ${job_name}"
  local line2="Job id: ${job_id}"
  local line3="Submitted: ${submitted}"
  local line4="Ended: ${ended}"

  local max_len=${#line1}
  if (( ${#line2} > max_len )); then max_len=${#line2}; fi
  if (( ${#line3} > max_len )); then max_len=${#line3}; fi
  if (( ${#line4} > max_len )); then max_len=${#line4}; fi

  local width=$(( max_len + 4 ))
  local border
  printf -v border '%*s' "$width" ''
  border=${border// /=}

  echo "$border"
  printf "= %-${max_len}s =\n" "$line1"
  printf "= %-${max_len}s =\n" "$line2"
  printf "= %-${max_len}s =\n" "$line3"
  printf "= %-${max_len}s =\n" "$line4"
  echo "$border"
}

aggregate_when_ready() {
  local job_id="$1"
  local job_name="$2"
  local submit_time="$3"
  local logs_dir="cluster_logs"

  # Wait for job to leave the queue (finished/cancelled/failed)
  while [[ -n "$(squeue -j "$job_id" -h 2>/dev/null)" ]]; do
    sleep 2
  done

  local out_file="${logs_dir}/${job_name}_${job_id}.out"
  local err_file="${logs_dir}/${job_name}_${job_id}.err"

  local end_time
  end_time=$(date '+%Y-%m-%d %H:%M:%S')

  # Wait up to 60s for log files to appear (scheduler flush)
  local waited=0
  while [[ $waited -lt 60 && ! -f "$out_file" && ! -f "$err_file" ]]; do
    sleep 2
    waited=$(( waited + 2 ))
  done

  # Prepare temp files to avoid interleaving when multiple jobs complete simultaneously
  local tmp_out
  local tmp_err
  tmp_out=$(mktemp)
  tmp_err=$(mktemp)

  {
    print_eq_box "$job_name" "$job_id" "$submit_time" "$end_time"
    if [[ -f "$out_file" ]]; then
      cat "$out_file"
    else
      echo "(output file not found)"
    fi
    echo ""
  } > "$tmp_out"

  {
    print_eq_box "$job_name" "$job_id" "$submit_time" "$end_time"
    if [[ -f "$err_file" ]]; then
      cat "$err_file"
    else
      echo "(error file not found)"
    fi
    echo ""
  } > "$tmp_err"

  # Append atomically as much as possible
  cat "$tmp_out" >> "$AGG_OUT_FILE"
  cat "$tmp_err" >> "$AGG_ERR_FILE"
  rm -f "$tmp_out" "$tmp_err"
}

get_job_count() {
  # Count current user's jobs (running + pending). -h hides header so we count only jobs.
  squeue -u "$USER" -h | wc -l | awk '{print $1}'
}

submit_job() {
  local job_file="$1"
  local script_path="cluster_jobs/$job_file"

  if [[ ! -f "$script_path" ]]; then
    echo "Skip: job script not found: $script_path"
    return 0
  fi

  echo "Submitting: $script_path"
  # Use --parsable to output job id only
  local job_id
  if ! job_id=$(sbatch --parsable "$script_path"); then
    echo "✗ Failed to submit $script_path" >&2
    return 1
  fi

  echo "✓ Submitted as Job ID: $job_id"
  mv -f "$script_path" cluster_jobs/done/

  # Derive job_name from script filename: instance_<graph>.sh -> <graph>
  local base_name
  base_name=$(basename "$job_file")
  local job_name
  job_name="${base_name#instance_}"
  job_name="${job_name%.sh}"

  # Record submit time (human-readable)
  local submit_time
  submit_time=$(date '+%Y-%m-%d %H:%M:%S')

  # Aggregate outputs/errors when the job finishes (background)
  aggregate_when_ready "$job_id" "$job_name" "$submit_time" &
}

# Read the job list line by line, preserving order
total=0
while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  # Remove Windows CR and trim whitespace (tabs/spaces)
  line="${raw_line//$'\r'/}"
  # Trim leading whitespace
  line="${line#"${line%%[![:space:]]*}"}"
  # Trim trailing whitespace
  line="${line%"${line##*[![:space:]]}"}"

  # Skip blanks and comments
  [[ -z "$line" ]] && continue
  [[ "$line" =~ ^# ]] && continue

  # Wait until we are below the job limit
  while true; do
    current_jobs=$(get_job_count)
    if (( current_jobs < JOB_LIMIT )); then
      break
    fi
    echo "At limit ($current_jobs >= $JOB_LIMIT). Waiting..."
    sleep 1
  done

  if submit_job "$line"; then
    total=$(( total + 1 ))
  else
    echo "Error submitting job: $line. Continuing to next."
  fi
done < "$JOB_LIST_FILE"

echo ""
echo "============================================"
echo " ALL ELIGIBLE JOBS SUBMITTED (as capacity allowed)"
echo "============================================"
echo " Total submitted from list: $total"
echo " Monitor with: squeue -u $USER"
echo " Cancel example:  scancel <job_id>"


