#!/bin/bash
#SBATCH --job-name=DSN_Gset
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=hickory
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:40g:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/DSN_Gset_%j.out
#SBATCH --error=cluster_logs/DSN_Gset_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: DSN"
echo "Dataset: Gset"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: Gset, -----------------------------------------------------------"
python main.py \
  --solver DSN \
  --dataset Gset \

if [ $? -eq 0 ]; then
  echo "✓ DSN on Gset SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ DSN on Gset FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
