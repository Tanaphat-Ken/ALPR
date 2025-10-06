#!/bin/bash
#SBATCH --job-name=alpr_test
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_test_output_%j.out
#SBATCH --error=slurm_test_error_%j.err

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
echo "Starting quick test experiments at $(date)"

# Run the experiments with very small sample sizes for quick testing
python train/run_experiments.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --output-root outputs/test_grid \
    --num-workers 2 \
    --per-device-eval-batch-size 16 \
    --per-device-train-batch-size 16 \
    --learning-rate 5e-5 \
    --num-train-epochs 2 \
    --eval-num-beams 3 \
    --eval-batch-size 16 \
    --max-train-samples 50 \
    --max-eval-samples 20 \
    --eval-max-samples 30 \
    --continue-on-error \
    --experiments exp3_kkatiz_clean exp5_openthaigpt_aug

echo "Test experiments completed at $(date)"