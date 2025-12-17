#!/bin/bash
#SBATCH --job-name=SPONGE_barabasi_albert
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=beech
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/SPONGE_barabasi_albert_%j.out
#SBATCH --error=cluster_logs/SPONGE_barabasi_albert_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: SPONGE"
echo "Dataset: barabasi_albert"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: barabasi_albert, -----------------------------------------------------------"
python main.py \
  --solver SPONGE \
  --dataset barabasi_albert \

if [ $? -eq 0 ]; then
  echo "✓ SPONGE on barabasi_albert SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ SPONGE on barabasi_albert FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
