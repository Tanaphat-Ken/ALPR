#!/bin/bash
#SBATCH --job-name=exp2_v1_medium
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-03:00:00
#SBATCH --output=slurm_exp2_v1_medium_%j.out
#SBATCH --error=slurm_exp2_v1_medium_%j.err

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Activate virtual environment
source .venv_linux/bin/activate

echo "GPU Information for exp2_v1_medium:"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting exp2_v1_medium (medium augmentation) at $(date)"

# Run V1 model with MEDIUM augmentation (faster than heavy)
python train/train_trocr.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-dir outputs/grid/exp2_v1_medium \
    --model-id openthaigpt/thai-trocr \
    --model-path models/weights/charactor_reader.pth \
    --augment medium \
    --fp16 \
    --predict-province \
    --province-format code \
    --num-workers 2 \
    --per-device-eval-batch-size 32 \
    --per-device-train-batch-size 32 \
    --learning-rate 3e-5 \
    --num-train-epochs 6 \
    --eval-strategy epoch \
    --save-strategy epoch

echo "exp2_v1_medium completed at $(date)"