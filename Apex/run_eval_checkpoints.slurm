#!/bin/bash
#SBATCH --job-name=eval_checkpoints
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --output=slurm_eval_checkpoints_%j.out
#SBATCH --error=slurm_eval_checkpoints_%j.err

# Set default experiment if not provided as argument
EXPERIMENT_DIR=${1:-"outputs/grid/exp1_v1_clean"}

echo "Starting checkpoint evaluation at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Evaluating experiment: $EXPERIMENT_DIR"

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Load environment
source .venv_linux/bin/activate

# Print system info
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo "=== Running checkpoint evaluation ==="

# Run checkpoint evaluation
python train/eval_checkpoints.py \
    --experiment-dir "$EXPERIMENT_DIR" \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --split test \
    --overwrite

echo "=== Checkpoint evaluation completed at $(date) ==="