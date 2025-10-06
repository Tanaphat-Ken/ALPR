#!/bin/bash
#SBATCH --job-name=exp1_v1_clean
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm_exp1_v1_clean_%j.out
#SBATCH --error=slurm_exp1_v1_clean_%j.err

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Activate virtual environment
source .venv_linux/bin/activate

# Check GPU availability
echo "GPU Information for exp1_v1_clean:"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting exp1_v1_clean at $(date)"

# Run single experiment with optimized parameters
python train/run_experiments.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-root outputs/grid \
    --num-workers 4 \
    --per-device-eval-batch-size 48 \
    --per-device-train-batch-size 48 \
    --learning-rate 3e-5 \
    --num-train-epochs 6 \
    --eval-strategy epoch \
    --save-strategy epoch \
    --eval-num-beams 3 \
    --eval-batch-size 48 \
    --continue-on-error \
    --experiments exp1_v1_clean

echo "exp1_v1_clean completed at $(date)"