#!/bin/bash
#SBATCH --job-name=SpecAnn_Gset
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=beech
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/SpecAnn_Gset_%j.out
#SBATCH --error=cluster_logs/SpecAnn_Gset_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: SpecAnn"
echo "Dataset: Gset"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: Gset, -----------------------------------------------------------"
python main.py \
  --solver SpecAnn \
  --dataset Gset \

if [ $? -eq 0 ]; then
  echo "✓ SpecAnn on Gset SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ SpecAnn on Gset FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
