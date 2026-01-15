#!/bin/bash
#SBATCH --job-name=GUROBI_15_K15
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:10:00
#SBATCH --output=cluster_logs/GUROBI_15_K15_%j.out
#SBATCH --error=cluster_logs/GUROBI_15_K15_%j.err

module load anaconda
source ~/.bashrc
conda activate spec-ising

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running dataset: hand_test2"
echo "Instance: K15.txt"
echo "Solver: GUROBI_15"

START_TIME=$(date +%s)
TOTAL=0

echo "[INSTANCE] Solver: GUROBI_15, Dataset: hand_test2, Instance: K15.txt"
python main.py \
  --solver GUROBI_15 \
  --dataset hand_test2 \
  --instance K15.txt \

if [ $? -eq 0 ]; then
  echo "✓ GUROBI_15 on hand_test2/K15.txt SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ GUROBI_15 on hand_test2/K15.txt FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
