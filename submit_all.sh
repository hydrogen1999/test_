#!/bin/bash
# Submit all generated SLURM job scripts

MODE="single" #put "multiple" in case your scripts are generated in multiple mode

if [[ "$MODE" != "single" && "$MODE" != "multiple" ]]; then
  echo "Usage: ./submit_all.sh [single|multiple]"
  exit 1
fi

echo "============================================"
echo " SUBMITTING ISING JOBS IN MODE: $MODE"
echo "============================================"

cd "$(dirname "$0")"

if [[ "$MODE" == "single" ]]; then
  JOB_SCRIPTS=$(find ./cluster_jobs -maxdepth 1 -name '*.sh' | sort)
else
  JOB_SCRIPTS=$(find ./cluster_jobs -maxdepth 1 -name '*_*.sh' ! -name '*_all.sh' | sort)
fi

TOTAL=0
JOB_IDS=()

for script in $JOB_SCRIPTS; do
  echo "Submitting: $script"
  JOB_ID=$(sbatch --parsable "$script")
  if [ $? -eq 0 ]; then
    echo "✓ Submitted as Job ID: $JOB_ID"
    JOB_IDS+=("$JOB_ID")
    TOTAL=$((TOTAL + 1))
  else
    echo "✗ Failed to submit $script"
  fi
done

echo ""
echo "============================================"
echo " ALL JOBS SUBMITTED"
echo "============================================"
echo " Total jobs: $TOTAL"
echo " Job IDs: ${JOB_IDS[*]}"
echo ""
echo " Monitor with: squeue -u $USER"
echo " Cancel with:  scancel ${JOB_IDS[*]}"
