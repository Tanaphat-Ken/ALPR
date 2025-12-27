#!/bin/bash
#SBATCH --job-name=alpr_micro_test
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:30:00
#SBATCH --output=slurm_micro_test_output_%j.out
#SBATCH --error=slurm_micro_test_error_%j.err

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Activate virtual environment
source .venv_linux/bin/activate

echo "GPU Information:"
nvidia-smi

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting micro test experiments at $(date)"

# Ultra-minimal test with just a few samples
python train/run_experiments.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-root outputs/micro_test \
    --num-workers 2 \
    --per-device-eval-batch-size 4 \
    --per-device-train-batch-size 4 \
    --learning-rate 1e-4 \
    --num-train-epochs 1 \
    --eval-num-beams 1 \
    --eval-batch-size 4 \
    --max-train-samples 10 \
    --max-eval-samples 5 \
    --eval-max-samples 8 \
    --continue-on-error \
    --experiments exp3_kkatiz_clean

echo "Micro test experiment completed at $(date)"