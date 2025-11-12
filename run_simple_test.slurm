#!/bin/bash
#SBATCH --job-name=exp_simple
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=slurm_simple_%j.out
#SBATCH --error=slurm_simple_%j.err

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Activate virtual environment
source .venv_linux/bin/activate

echo "Starting simple test (no province prediction) at $(date)"

# Simple test - just license plate recognition without province
python train/train_trocr.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-dir outputs/test_simple \
    --model-id kkatiz/thai-trocr-thaigov-v2 \
    --augment none \
    --fp16 \
    --num-workers 0 \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 8 \
    --learning-rate 5e-5 \
    --num-train-epochs 1 \
    --eval-strategy epoch \
    --save-strategy epoch \
    --logging-steps 5 \
    --max-train-samples 50 \
    --max-eval-samples 10

echo "Simple test completed at $(date)"