#!/bin/bash
#SBATCH --job-name=QQA8_barabasi_albert_selective_QQA
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=beech
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/QQA8_barabasi_albert_selective_QQA_%j.out
#SBATCH --error=cluster_logs/QQA8_barabasi_albert_selective_QQA_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: QQA8"
echo "Dataset: barabasi_albert_selective_QQA"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: barabasi_albert_selective_QQA, -----------------------------------------------------------"
python main.py \
  --solver QQA8 \
  --dataset barabasi_albert_selective_QQA \
  --config QQA \

if [ $? -eq 0 ]; then
  echo "✓ QQA8 on barabasi_albert_selective_QQA SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ QQA8 on barabasi_albert_selective_QQA FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
