#!/bin/bash
#SBATCH --job-name=plate_only_improved
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm_plate_only_improved_%j.out
#SBATCH --error=slurm_plate_only_improved_%j.err

echo "Starting plate-only improved training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Load environment
source .venv_linux/bin/activate

# Print Python and package versions
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo "=== Running PLATE-ONLY training (no province prediction) ==="

# Run training without province prediction and with improved parameters
python train/train_trocr.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --model-id openthaigpt/thai-trocr \
    --output-dir outputs/plate_only_improved \
    --augment light \
    --num-train-epochs 8 \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 16 \
    --learning-rate 1e-5 \
    --warmup-steps 200 \
    --weight-decay 0.01 \
    --eval-strategy epoch \
    --save-strategy epoch \
    --fp16 \
    --gradient-accumulation-steps 2 \
    --logging-steps 50 \
    --eval-steps 500 \
    --save-steps 500 \
    --generation-max-length 16 \
    --seed 42

echo "=== Training completed at $(date) ==="

# Only run evaluation if training was successful
if [ -d "outputs/plate_only_improved" ] && [ -f "outputs/plate_only_improved/pytorch_model.bin" ]; then
    echo "=== Running evaluation on test set ==="
    
    # Evaluate on test set
    python train/eval_trocr.py \
        --csv data/8000/8000.csv \
        --data-root data/8000 \
        --model-id openthaigpt/thai-trocr \
        --model-path outputs/plate_only_improved \
        --split test \
        --batch-size 16 \
        --num-beams 5 \
        --save-results outputs/plate_only_improved/eval_test.json
else
    echo "=== Training failed or incomplete, skipping evaluation ==="
fi

echo "=== All completed at $(date) ==="