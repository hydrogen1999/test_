#!/bin/bash
#SBATCH --job-name=QQA_erdos_renyi
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=beech
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/QQA_erdos_renyi_%j.out
#SBATCH --error=cluster_logs/QQA_erdos_renyi_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: QQA"
echo "Dataset: erdos_renyi"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: erdos_renyi, -----------------------------------------------------------"
python main.py \
  --solver QQA \
  --dataset erdos_renyi \

if [ $? -eq 0 ]; then
  echo "✓ QQA on erdos_renyi SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ QQA on erdos_renyi FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
