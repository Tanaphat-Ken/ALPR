#!/bin/bash

# Interactive APEX execution script
# This allocates resources and runs the experiment interactively

echo "Requesting interactive APEX session with A100 GPU..."
echo "This will allocate: 1x A100 GPU, 16 CPUs, 64GB RAM for 3 days"

srun --qos=high --partition=batch --gres=gpu:a100:1 --cpus-per-task=16 --mem=64G --time=3-5:00:00 --pty bash -c "
    cd /home/tsomboo/ALPR/plate_recognizer && \
    source .venv_linux/bin/activate && \
    echo 'GPU Information:' && \
    nvidia-smi && \
    echo 'Starting experiments...' && \
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
"