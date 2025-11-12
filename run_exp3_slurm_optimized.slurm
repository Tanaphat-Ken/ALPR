#!/bin/bash
#SBATCH --job-name=exp3_slurm_opt
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=slurm_exp3_opt_%j.out
#SBATCH --error=slurm_exp3_opt_%j.err
#SBATCH --exclusive

# SLURM optimizations
export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Activate virtual environment
source .venv_linux/bin/activate

# Set GPU performance mode
nvidia-smi -pm 1
nvidia-smi -ac 1215,1410  # Set memory and GPU clocks to max

echo "=== System Information ==="
echo "GPU Information:"
nvidia-smi
echo "CPU Information:"
lscpu | grep "Model name\|CPU(s)\|Thread(s) per core"
echo "Memory Information:"  
free -h
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Starting exp3_slurm_opt at $(date)"

# Optimized training with SLURM-specific settings
python train/train_trocr.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-dir outputs/grid/exp3_kkatiz_optimized \
    --model-id kkatiz/thai-trocr-thaigov-v2 \
    --augment none \
    --fp16 \
    --predict-province \
    --province-format code \
    --num-workers 0 \
    --per-device-train-batch-size 24 \
    --per-device-eval-batch-size 24 \
    --gradient-accumulation-steps 1 \
    --learning-rate 5e-5 \
    --num-train-epochs 4 \
    --eval-strategy epoch \
    --save-strategy epoch \
    --logging-steps 10 \
    --warmup-steps 50 \
    --weight-decay 0.01

echo "exp3_slurm_opt completed at $(date)"