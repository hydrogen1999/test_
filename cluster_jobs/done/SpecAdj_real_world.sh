#!/bin/bash
#SBATCH --job-name=SpecAdj_real_world
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=hickory
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:40g:1
#SBATCH --time=2-00:00:00
#SBATCH --output=cluster_logs/SpecAdj_real_world_%j.out
#SBATCH --error=cluster_logs/SpecAdj_real_world_%j.err

module load anaconda
source ~/.bashrc
conda activate ising-gpu

cd $SLURM_SUBMIT_DIR
export PYTHONPATH=$PWD:$PYTHONPATH

echo "Running solver: SpecAdj"
echo "Dataset: real_world"

START_TIME=$(date +%s)
TOTAL=0

echo "[GENERAL] Dataset: real_world, -----------------------------------------------------------"
python main.py \
  --solver SpecAdj \
  --dataset real_world \

if [ $? -eq 0 ]; then
  echo "✓ SpecAdj on real_world SUCCESS"
  TOTAL=$((TOTAL + 1))
else
  echo "✗ SpecAdj on real_world FAILED"
  exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Completed $TOTAL experiments in $DURATION seconds"
