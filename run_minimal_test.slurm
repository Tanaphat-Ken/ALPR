#!/bin/bash
#SBATCH --job-name=exp_minimal_test
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=slurm_minimal_test_%j.out
#SBATCH --error=slurm_minimal_test_%j.err

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Activate virtual environment
source .venv_linux/bin/activate

echo "GPU Information for minimal test:"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting minimal test at $(date)"

# Minimal test with very small dataset to check speed
python train/train_trocr.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-dir outputs/test_minimal \
    --model-id kkatiz/thai-trocr-thaigov-v2 \
    --augment none \
    --fp16 \
    --predict-province \
    --province-format code \
    --num-workers 0 \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 8 \
    --gradient-accumulation-steps 1 \
    --learning-rate 5e-5 \
    --num-train-epochs 1 \
    --max-train-samples 100 \
    --max-eval-samples 20 \
    --eval-strategy epoch \
    --save-strategy epoch \
    --logging-steps 5

echo "Minimal test completed at $(date)"