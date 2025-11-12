#!/bin/bash
#SBATCH --job-name=exp3_fast
#SBATCH --qos=high
#SBATCH --partition=batch
#SBTML:function_calls>
<invoke name="create_file">
<parameter name="content">#!/bin/bash
#SBATCH --job-name=exp3_fast
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_exp3_fast_%j.out
#SBATCH --error=slurm_exp3_fast_%j.err

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Activate virtual environment
source .venv_linux/bin/activate

echo "GPU Information for exp3_fast:"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting exp3_fast (ultra-optimized) at $(date)"

# Ultra-optimized direct training call with gradient accumulation
python train/train_trocr.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-dir outputs/grid/exp3_kkatiz_clean \
    --model-id kkatiz/thai-trocr-thaigov-v2 \
    --augment none \
    --fp16 \
    --predict-province \
    --province-format code \
    --num-workers 0 \
    --per-device-train-batch-size 16 \
    --per-device-eval-batch-size 16 \
    --gradient-accumulation-steps 2 \
    --learning-rate 5e-5 \
    --num-train-epochs 4 \
    --eval-strategy epoch \
    --save-strategy epoch \
    --logging-steps 10 \
    --warmup-steps 100 \
    --weight-decay 0.01 \
    --dataloader-pin-memory false \
    --dataloader-num-workers 0

echo "exp3_fast completed at $(date)"