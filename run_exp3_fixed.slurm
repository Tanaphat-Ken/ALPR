#!/bin/bash
#SBATCH --job-name=exp3_fixed
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_exp3_fixed_%j.out
#SBATCH --error=slurm_exp3_fixed_%j.err

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Activate virtual environment
source .venv_linux/bin/activate

echo "GPU Information for exp3_fixed:"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting exp3_fixed (without province prediction to avoid tokenizer issues) at $(date)"

# Run without province prediction to avoid tokenizer/embedding issues
python train/train_trocr.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-dir outputs/grid/exp3_kkatiz_fixed \
    --model-id kkatiz/thai-trocr-thaigov-v2 \
    --augment none \
    --fp16 \
    --num-workers 0 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 16 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --num-train-epochs 2 \
    --eval-strategy epoch \
    --save-strategy epoch \
    --logging-steps 10 \
    --max-train-samples 1000 \
    --max-eval-samples 200

echo "exp3_fixed completed at $(date)"