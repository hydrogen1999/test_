#!/usr/bin/env python3
"""
SLURM Job Script Generator for Ising Solver Benchmarking

Generates one job per instance per dataset for each solver.
"""

import os
import argparse
from pathlib import Path

SOLVERS = [
    'GUROBI_15',
    'SA_01',
]

DATASETS = ['hand_test2', 'hand_test_gset']

SLURM_CONFIG = {
    'partition': 'cpu',               # Use CPU partition
    'nodes': 1,                       # Only 1 node
    'ntasks': 1,                      # Single task
    'cpus': 1,                        # Adjust as needed
    'memory': '8G',                  # Lower memory if GPU not used
    'time': '02:10:00',
    'job_name_prefix': 'ising_cpu',
    'output_dir': 'cluster_logs'
}

# For SA only
INSTANCE_MEMORY = {
    '64': '2G',
    '128': '2G',
    '256': '2G',
    '512': '2G',
    '1024': '2G',
    '2048': '2G',
    '4096': '4G',
    '8192': '4G',
    '16384': '4G',
    '32768': '4G',
    '65536': '4G',
    '131072': '8G',
    '262144': '16G',
    '524288': '32G',
    '1048576': '64G',
}

#INSTANCE_MEMORY = {
#    '64': '4G',
#    '128': '4G',
#    '256': '4G',
#    '512': '8G',
#    '1024': '8G',
#    '2048': '16G',
#    '4096': '16G',
#    '8192': '32G',
#    '16384': '32G',
#    '32768': '64G',
#    '65536': '128G',
#    '131072': '256G',
#    '262144': '256G',
#    '524288': '256G',
#    '1048576': '256G',
#}

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --nodes={nodes}
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={logs_dir}/{job_name}_%j.out
#SBATCH --error={logs_dir}/{job_name}_%j.err

module load anaconda
source ~/.bashrc
conda activate {conda_env}

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running dataset: {dataset}"
echo "Instance: {instance}"
echo "Solver: {solver}"

START_TIME=$(date +%s)
TOTAL=0

echo "[INSTANCE] Solver: {solver}, Dataset: {dataset}, Instance: {instance}"
python main.py \\
  --solver {solver} \\
  --dataset {dataset} \\
  --instance {instance} \\

if [ $? -eq 0 ]; then
  echo "✓ {solver} on {dataset}/{instance} SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ {solver} on {dataset}/{instance} FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
"""


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    os.makedirs(path, exist_ok=True)


def get_instance_size(instance_name: str) -> str:
    """Extract instance size from filename (e.g., ba_1024_1_0.txt -> 1024)."""
    parts = instance_name.split('_')
    if len(parts) >= 2:
        return parts[1]  # Second part should be the size
    return None


def get_memory_for_instance(instance_name: str, default_memory: str) -> str:
    """Get memory allocation for instance based on its size."""
    size = get_instance_size(instance_name)
    if size and size in INSTANCE_MEMORY:
        return INSTANCE_MEMORY[size]
    return default_memory  # Use provided default


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM submit scripts for Ising solver benchmarking.")
    parser.add_argument("--datasets-dir", default="datasets", help="Directory containing dataset subdirectories")
    parser.add_argument("--jobs-dir", default="cluster_jobs", help="Output directory for generated submit scripts")
    parser.add_argument("--logs-dir", default="cluster_logs", help="Logs directory used by SBATCH output/error")
    parser.add_argument("--conda-env", default="spec-ising", help="Conda environment to activate")

    # SBATCH parameters
    parser.add_argument("--partition", default=SLURM_CONFIG['partition'])
    parser.add_argument("--nodes", type=int, default=SLURM_CONFIG['nodes'])
    parser.add_argument("--ntasks", type=int, default=SLURM_CONFIG['ntasks'])
    parser.add_argument("--cpus", type=int, default=SLURM_CONFIG['cpus'])
    parser.add_argument("--mem", default=SLURM_CONFIG['memory'])
    parser.add_argument("--time", default=SLURM_CONFIG['time'])

    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(root_dir, args.datasets_dir)
    jobs_dir = os.path.join(root_dir, args.jobs_dir)
    logs_dir = os.path.join(root_dir, args.logs_dir)

    ensure_dir(jobs_dir)
    ensure_dir(logs_dir)

    created_scripts = []

    # Iterate over datasets
    for dataset in DATASETS:
        dataset_path = os.path.join(datasets_dir, dataset)
        if not os.path.exists(dataset_path):
            print(f"! Skipping {dataset}: {dataset_path} not found")
            continue

        # Collect instance files (*.txt, *.bin, *.bin.gz)
        instance_files = []
        for fname in sorted(os.listdir(dataset_path)):
            if (fname.endswith('.txt') or fname.endswith('.bin') or fname.endswith('.bin.gz')) and os.path.isfile(os.path.join(dataset_path, fname)):
                instance_files.append(fname)

        if not instance_files:
            print(f"! No instances found for {dataset}")
            continue

        # Generate scripts for each solver and instance combination
        for instance in instance_files:
            # Remove extension(s) - handle both .txt and .bin.gz cases
            if instance.endswith('.bin.gz'):
                instance_base = instance[:-7]  # Remove .bin.gz
            elif instance.endswith('.bin'):
                instance_base = instance[:-4]  # Remove .bin
            else:
                instance_base = os.path.splitext(instance)[0]  # Remove .txt extension
            
            # Get memory allocation based on instance size
            instance_memory = get_memory_for_instance(instance, args.mem)
            
            for solver in SOLVERS:
                job_name = f"{solver}_{instance_base}"
                script_name = f"{solver}_{instance_base}.sh"
                script_path = os.path.join(jobs_dir, script_name)

                content = TEMPLATE.format(
                    job_name=job_name,
                    partition=args.partition,
                    nodes=args.nodes,
                    ntasks=args.ntasks,
                    cpus=args.cpus,
                    mem=instance_memory,  # Use dynamic memory allocation
                    time=args.time,
                    logs_dir=args.logs_dir,
                    conda_env=args.conda_env,
                    dataset=dataset,
                    instance=instance,
                    solver=solver,
                )

                with open(script_path, 'w') as f:
                    f.write(content)

                # Make the script executable
                os.chmod(script_path, 0o755)

                # Track created script name for job list
                created_scripts.append(script_name)

                print(f"✓ {script_name} (memory: {instance_memory})")

    # Write job list in project root (one script per line, in priority order)
    job_list_path = os.path.join(root_dir, f'job_list_{args.conda_env}.txt')
    with open(job_list_path, 'w') as job_list_file:
        job_list_file.write("\n".join(created_scripts) + ("\n" if created_scripts else ""))

    print(f"Generated {len(created_scripts)} submit scripts in '{jobs_dir}' for datasets in '{datasets_dir}'.")


if __name__ == "__main__":
    main()
