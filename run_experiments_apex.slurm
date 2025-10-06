#!/bin/bash
#SBATCH --job-name=alpr_experiments
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-05:00:00
#SBATCH --output=slurm_output_%j.out
#SBATCH --error=slurm_error_%j.err

# Load any necessary modules (if required by your system)
# module load python/3.8
# module load cuda/11.8

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Activate virtual environment
source .venv_linux/bin/activate

# Check GPU availability
echo "GPU Information:"
nvidia-smi

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Starting experiments at $(date)"

# Run the experiments with APEX-optimized parameters
python train/run_experiments.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-root outputs/grid \
    --num-workers 8 \
    --per-device-eval-batch-size 16 \
    --per-device-train-batch-size 16 \
    --learning-rate 2e-5 \
    --num-train-epochs 5 \
    --eval-num-beams 5 \
    --eval-batch-size 16 \
    --continue-on-error

echo "Experiments completed at $(date)"