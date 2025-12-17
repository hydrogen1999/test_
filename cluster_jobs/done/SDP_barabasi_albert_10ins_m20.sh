#!/bin/bash
#SBATCH --job-name=SDP_barabasi_albert_10ins_m20
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=beech
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/SDP_barabasi_albert_10ins_m20_%j.out
#SBATCH --error=cluster_logs/SDP_barabasi_albert_10ins_m20_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: SDP"
echo "Dataset: barabasi_albert_10ins_m20"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: barabasi_albert_10ins_m20, -----------------------------------------------------------"
python main.py \
  --solver SDP \
  --dataset ba2 \

if [ $? -eq 0 ]; then
  echo "✓ SDP on barabasi_albert_10ins_m20 SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ SDP on barabasi_albert_10ins_m20 FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
