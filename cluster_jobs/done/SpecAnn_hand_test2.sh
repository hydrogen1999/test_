#!/bin/bash
#SBATCH --job-name=SpecAnn_hand_test2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=hickory
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:40g:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/SpecAnn_hand_test2_%j.out
#SBATCH --error=cluster_logs/SpecAnn_hand_test2_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: SpecAnn"
echo "Dataset: hand_test2"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: hand_test2, -----------------------------------------------------------"
python main.py \
  --solver SpecAnn \
  --dataset hand_test2 \

if [ $? -eq 0 ]; then
  echo "✓ SpecAnn on hand_test2 SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ SpecAnn on hand_test2 FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
