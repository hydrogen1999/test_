#!/bin/bash
#SBATCH --job-name=SDP_hand_test2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=beech
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/SDP_hand_test2_%j.out
#SBATCH --error=cluster_logs/SDP_hand_test2_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: SDP"
echo "Dataset: hand_test2"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: hand_test2, -----------------------------------------------------------"
python main.py \
  --solver SDP \
  --dataset hand_test2 \

if [ $? -eq 0 ]; then
  echo "✓ SDP on hand_test2 SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ SDP on hand_test2 FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
