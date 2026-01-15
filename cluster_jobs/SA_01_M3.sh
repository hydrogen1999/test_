#!/bin/bash
#SBATCH --job-name=SA_01_M3
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:10:00
#SBATCH --output=cluster_logs/SA_01_M3_%j.out
#SBATCH --error=cluster_logs/SA_01_M3_%j.err

module load anaconda
source ~/.bashrc
conda activate spec-ising

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running dataset: hand_test_gset"
echo "Instance: M3.txt"
echo "Solver: SA_01"

START_TIME=$(date +%s)
TOTAL=0

echo "[INSTANCE] Solver: SA_01, Dataset: hand_test_gset, Instance: M3.txt"
python main.py \
  --solver SA_01 \
  --dataset hand_test_gset \
  --instance M3.txt \

if [ $? -eq 0 ]; then
  echo "✓ SA_01 on hand_test_gset/M3.txt SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ SA_01 on hand_test_gset/M3.txt FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
