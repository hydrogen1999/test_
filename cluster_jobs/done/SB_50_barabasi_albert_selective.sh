#!/bin/bash
#SBATCH --job-name=SB_50_barabasi_albert_selective
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=beech
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/SB_50_barabasi_albert_selective_%j.out
#SBATCH --error=cluster_logs/SB_50_barabasi_albert_selective_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: SB_50"
echo "Dataset: barabasi_albert_selective"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: barabasi_albert_selective, -----------------------------------------------------------"
python main.py \
  --solver SB_50 \
  --dataset barabasi_albert_selective \
  --config selective \

if [ $? -eq 0 ]; then
  echo "✓ SB_50 on barabasi_albert_selective SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ SB_50 on barabasi_albert_selective FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
