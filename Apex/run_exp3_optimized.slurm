#!/bin/bash
#SBATCH --job-name=exp3_kkatiz_clean_opt
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-02:00:00
#SBATCH --output=slurm_exp3_kkatiz_clean_opt_%j.out
#SBATCH --error=slurm_exp3_kkatiz_clean_opt_%j.err

echo "Starting optimized exp3_kkatiz_clean training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Load environment
source .venv_linux/bin/activate

# Print system info
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo "=== Running EXP3 KKATIZ CLEAN with optimized settings ==="

# Run using the optimized run_experiments.py script with A100-optimized settings
python train/run_experiments.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --experiments exp3_kkatiz_clean \
    --eval-only-at-end \
    --num-train-epochs 5 \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 16 \
    --learning-rate 1e-5 \
    --num-workers 8 \
    --eval-batch-size 16 \
    --eval-num-beams 5 \
    --continue-on-error

echo "=== EXP3 KKATIZ CLEAN completed at $(date) ==="