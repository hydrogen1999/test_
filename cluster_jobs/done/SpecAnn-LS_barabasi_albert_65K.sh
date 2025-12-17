#!/bin/bash
#SBATCH --job-name=SpecAnn-LS_barabasi_albert_65K
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=beech
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/SpecAnn-LS_barabasi_albert_65K_%j.out
#SBATCH --error=cluster_logs/SpecAnn-LS_barabasi_albert_65K_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: SpecAnn-LS"
echo "Dataset: barabasi_albert_65K"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: barabasi_albert_65K, -----------------------------------------------------------"
python main.py \
  --solver SpecAnn-LS \
  --dataset barabasi_albert_65K \

if [ $? -eq 0 ]; then
  echo "✓ SpecAnn-LS on barabasi_albert_65K SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ SpecAnn-LS on barabasi_albert_65K FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
