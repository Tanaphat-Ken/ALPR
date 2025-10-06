#!/bin/bash
#SBATCH --job-name=exp5_openthaigpt_aug
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm_exp5_openthaigpt_aug_%j.out
#SBATCH --error=slurm_exp5_openthaigpt_aug_%j.err

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Activate virtual environment
source .venv_linux/bin/activate

# Check GPU availability
echo "GPU Information for exp5_openthaigpt_aug:"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting exp5_openthaigpt_aug at $(date)"

# Run single experiment with optimized parameters
python train/run_experiments.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-root outputs/grid \
    --num-workers 4 \
    --per-device-eval-batch-size 48 \
    --per-device-train-batch-size 48 \
    --learning-rate 3e-5 \
    --num-train-epochs 3 \
    --eval-num-beams 3 \
    --eval-batch-size 48 \
    --continue-on-error \
    --experiments exp5_openthaigpt_aug

echo "exp5_openthaigpt_aug completed at $(date)"