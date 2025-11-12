#!/bin/bash
#SBATCH --job-name=plate_high_quality
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=slurm_plate_high_quality_%j.out
#SBATCH --error=slurm_plate_high_quality_%j.err

echo "Starting high-quality plate training at $(date)"
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

echo "=== Running HIGH-QUALITY plate training (optimized for low CER) ==="

# High-quality training with optimal settings for Thai plate recognition
python train/train_trocr.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --model-id openthaigpt/thai-trocr \
    --output-dir outputs/plate_high_quality \
    --augment medium \
    --num-train-epochs 15 \
    --per-device-train-batch-size 4 \
    --per-device-eval-batch-size 8 \
    --learning-rate 5e-6 \
    --warmup-steps 300 \
    --weight-decay 0.01 \
    --eval-strategy epoch \
    --save-strategy epoch \
    --fp16 \
    --gradient-accumulation-steps 4 \
    --logging-steps 25 \
    --generation-max-length 20 \
    --seed 42

echo "=== High-quality training completed at $(date) ==="

# Only run evaluation if training was successful
if [ -d "outputs/plate_high_quality" ] && [ -f "outputs/plate_high_quality/pytorch_model.bin" ]; then
    echo "=== Running comprehensive evaluation ==="

    # Evaluate on test set with multiple beam sizes
    for beams in 1 3 5; do
        echo "Evaluating with num_beams=$beams"
        python train/eval_trocr.py \
            --csv data/8000/8000.csv \
            --data-root data/8000 \
            --model-id openthaigpt/thai-trocr \
            --model-path outputs/plate_high_quality \
            --split test \
            --batch-size 16 \
            --num-beams $beams \
            --save-results outputs/plate_high_quality/eval_test_beam${beams}.json
    done
else
    echo "=== Training failed or incomplete, skipping evaluation ==="
fi

echo "=== All evaluations completed at $(date) ==="