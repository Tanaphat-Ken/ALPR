#!/bin/bash
#SBATCH --job-name=kkatiz_optimized
#SBATCH --qos=high
#SBATCH --partition=batch
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-05:00:00
#SBATCH --output=slurm_kkatiz_optimized_%j.out
#SBATCH --error=slurm_kkatiz_optimized_%j.err

echo "Starting optimized kkatiz training at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Change to the project directory
cd /home/tsomboo/ALPR/plate_recognizer

# Load environment
source .venv_linux/bin/activate

echo "=== Running KKATIZ model training (proven for Thai government plates) ==="

# Use kkatiz model which is specifically trained for Thai government plates
python train/train_trocr.py \
    --csv data/8000/8000.csv \
    --data-root data/8000 \
    --model-id kkatiz/thai-trocr-thaigov-v2 \
    --output-dir outputs/kkatiz_optimized \
    --augment light \
    --num-train-epochs 10 \
    --per-device-train-batch-size 8 \
    --per-device-eval-batch-size 16 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --weight-decay 0.01 \
    --eval-strategy epoch \
    --save-strategy epoch \
    --fp16 \
    --gradient-accumulation-steps 2 \
    --logging-steps 50 \
    --generation-max-length 16 \
    --seed 42

echo "=== Kkatiz training completed at $(date) ==="

# Only run evaluation if training was successful
if [ -d "outputs/kkatiz_optimized" ] && [ -f "outputs/kkatiz_optimized/pytorch_model.bin" ]; then
    echo "=== Running evaluation ==="

    python train/eval_trocr.py \
        --csv data/8000/8000.csv \
        --data-root data/8000 \
        --model-id kkatiz/thai-trocr-thaigov-v2 \
        --model-path outputs/kkatiz_optimized \
        --split test \
        --batch-size 16 \
        --num-beams 5 \
        --save-results outputs/kkatiz_optimized/eval_test.json
else
    echo "=== Training failed or incomplete, skipping evaluation ==="
fi

echo "=== All completed at $(date) ==="