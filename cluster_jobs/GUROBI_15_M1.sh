#!/bin/bash
#SBATCH --job-name=GUROBI_15_M1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:10:00
#SBATCH --output=cluster_logs/GUROBI_15_M1_%j.out
#SBATCH --error=cluster_logs/GUROBI_15_M1_%j.err

module load anaconda
source ~/.bashrc
conda activate spec-ising

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running dataset: hand_test_gset"
echo "Instance: M1.txt"
echo "Solver: GUROBI_15"

START_TIME=$(date +%s)
TOTAL=0

echo "[INSTANCE] Solver: GUROBI_15, Dataset: hand_test_gset, Instance: M1.txt"
python main.py \
  --solver GUROBI_15 \
  --dataset hand_test_gset \
  --instance M1.txt \

if [ $? -eq 0 ]; then
  echo "✓ GUROBI_15 on hand_test_gset/M1.txt SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ GUROBI_15 on hand_test_gset/M1.txt FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
