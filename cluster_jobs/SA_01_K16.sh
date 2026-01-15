#!/bin/bash
#SBATCH --job-name=SA_01_K16
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:10:00
#SBATCH --output=cluster_logs/SA_01_K16_%j.out
#SBATCH --error=cluster_logs/SA_01_K16_%j.err

module load anaconda
source ~/.bashrc
conda activate spec-ising

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running dataset: hand_test2"
echo "Instance: K16.txt"
echo "Solver: SA_01"

START_TIME=$(date +%s)
TOTAL=0

echo "[INSTANCE] Solver: SA_01, Dataset: hand_test2, Instance: K16.txt"
python main.py \
  --solver SA_01 \
  --dataset hand_test2 \
  --instance K16.txt \

if [ $? -eq 0 ]; then
  echo "✓ SA_01 on hand_test2/K16.txt SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ SA_01 on hand_test2/K16.txt FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
