#!/usr/bin/env python3
"""
SLURM Job Script Generator for Ising Solver Benchmarking (GPU Version)

Generates one job per solver-dataset combination with GPU support.
"""

import os
import argparse
from pathlib import Path

SOLVERS = [
    #'',
    'SpecAdj',
    'SpecSignedLap',
    'SpecAnn',
    # 'SpecAnn-LS',
    #'SpecAnn-cold',
    #'SpecAnn-64',
    # 'QQA',
    #'SPONGE',
    #'SpecAnn-FastEig',
    #'SpecAnn-FastL-FastEig',
    #'SpecAnn-FastL',
    #'onlyLS',
    #'SpecAnn-NCV',
    #'SpecAnn-Trevisan',
    # 'SB_10',
    # 'SB_20',
    # 'SB_50',
    # 'SB_100',
    # 'SB_200',
    # 'SB_500',
    # 'SB_1000',
    # 'SB_2000',
    # 'SB_5000',
    # 'SB_10000',
    # 'SpecAnn_1',
    # 'SpecAnn_2',
    # 'SpecAnn_3',
    # 'SpecAnn_4',
    # 'SpecAnn_5',
    # 'SpecAnn_6',
    # 'SpecAnn_7',
    # 'SpecAnn_8',
    # 'SpecAnn_9',
    # 'SpecAnn_10',
    # 'QQA1',
    # 'QQA2',
    # 'QQA3',
    # 'QQA4',
    # 'QQA5',
    # 'QQA6',
    # 'QQA7',
    # 'QQA8',
    # 'QQA9',
    # 'QQA10',
]

#DATASETS = ['hand_test2']
#DATASETS = ['Gset']
DATASETS = ['real_world']
#DATASETS = ['barabasi_albert', 'barabasi_albert_10ins_m20']
#DATASETS = ['barabasi_albert_10ins_m20', 'erdos_renyi_10ins_m20']
# DATASETS = ['barabasi_albert_65K', 'erdos_renyi_10ins_65K', 'watts_strogatz_65K']
#DATASETS = ['barabasi_albert_10ins_m20', 'erdos_renyi_10ins_m20', 'barabasi_albert', 'erdos_renyi']
#DATASETS = ['erdos_renyi_10ins_m20', 'erdos_renyi']
#DATASETS = ['barabasi_albert_10ins_m20']
# DATASETS = ['barabasi_albert']
# DATASETS = ['barabasi_albert_selective_QQA']

SLURM_CONFIG = {
    'partition': 'gpu',               # Use GPU partition
    'nodelist': 'beech',
    'gres': 'gpu:1',                  # sinfo -N -o "%N %G %T"
    'time': '2-00:00:00',
    'nodes': 1,                       # Only 1 node
    'ntasks': 1,                      # Single task
    'memory': '128G',                 # GPU memory
    'job_name_prefix': 'ising_gpu',
    'output_dir': 'cluster_logs'
}


TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --nodelist={nodelist}
#SBATCH --ntasks={ntasks}
#SBATCH --mem={mem}
#SBATCH --gres={gres}
#SBATCH --time={time}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

module load anaconda
source ~/.bashrc
conda activate {conda_env}

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: {solver}"
echo "Dataset: {dataset}"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: {dataset}, -----------------------------------------------------------"
python main.py \\
  --solver {solver} \\
  --dataset {dataset} \\

if [ $? -eq 0 ]; then
  echo "✓ {solver} on {dataset} SUCCESS"
  TOTAL=$((TOTAL + {num_seeds}))
else
  echo "✗ {solver} on {dataset} FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
"""
#  --instance tmp.txt \\
#  --runtime_analysis runtime_tmp.prof \\


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)




def main():
    parser = argparse.ArgumentParser(description="Generate SLURM submit scripts for Ising solver benchmarking (GPU).")
    parser.add_argument("--datasets-dir", default="datasets", help="Directory containing dataset subdirectories")
    parser.add_argument("--jobs-dir", default="cluster_jobs", help="Output directory for generated submit scripts")
    parser.add_argument("--logs-dir", default="cluster_logs", help="Logs directory used by SBATCH output/error")
    parser.add_argument("--conda-env", default="ising-gpu", help="Conda environment to activate")

    # SBATCH parameters
    parser.add_argument("--partition", default=SLURM_CONFIG['partition'])
    parser.add_argument("--nodes", type=int, default=SLURM_CONFIG['nodes'])
    parser.add_argument("--nodelist", default=SLURM_CONFIG['nodelist'])
    parser.add_argument("--ntasks", type=int, default=SLURM_CONFIG['ntasks'])
    parser.add_argument("--mem", default=SLURM_CONFIG['memory'])
    parser.add_argument("--gres", default=SLURM_CONFIG['gres'])
    parser.add_argument("--time", default=SLURM_CONFIG['time'])

    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(root_dir, args.datasets_dir)
    jobs_dir = os.path.join(root_dir, args.jobs_dir)
    logs_dir = os.path.join(root_dir, args.logs_dir)

    ensure_dir(jobs_dir)
    ensure_dir(logs_dir)

    created_scripts = []

    # Generate scripts for each solver-dataset combination
    for solver in SOLVERS:
        for dataset in DATASETS:
            job_name = f"{solver}_{dataset}"
            script_name = f"{solver}_{dataset}.sh"
            script_path = os.path.join(jobs_dir, script_name)

            content = TEMPLATE.format(
                job_name=job_name,
                partition=args.partition,
                nodes=args.nodes,
                nodelist=args.nodelist,
                ntasks=args.ntasks,
                mem=args.mem,  # Use default memory for GPU jobs
                gres=args.gres,
                time=args.time,
                logs_dir=args.logs_dir,
                conda_env=args.conda_env,
                dataset=dataset,
                solver=solver,
                num_seeds=1,
            )

            with open(script_path, 'w') as f:
                f.write(content)

            # Make the script executable
            os.chmod(script_path, 0o755)

            # Track created script name for job list
            created_scripts.append(script_name)

            print(f"✓ {script_name} (memory: {args.mem})")

    # Write job list in project root (one script per line, in priority order)
    job_list_path = os.path.join(root_dir, f'job_list2.txt')
    with open(job_list_path, 'w') as job_list_file:
        job_list_file.write("\n".join(created_scripts) + ("\n" if created_scripts else ""))

    print(f"Generated {len(created_scripts)} submit scripts in '{jobs_dir}' for datasets in '{datasets_dir}'.")


if __name__ == "__main__":
    main()
