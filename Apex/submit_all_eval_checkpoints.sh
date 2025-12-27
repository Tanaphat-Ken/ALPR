#!/bin/bash

# Batch script to evaluate all experiments in the grid
echo "Submitting checkpoint evaluation jobs for all experiments..."

# List of experiments from the grid directory
EXPERIMENTS=(
    "outputs/grid/exp1_v1_clean"
    "outputs/grid/exp2_v1_aug" 
    "outputs/grid/exp3_kkatiz_clean"
    "outputs/grid/exp4_kkatiz_aug"
    "outputs/grid/exp5_openthaigpt_aug"
)

# Submit each experiment
for exp in "${EXPERIMENTS[@]}"; do
    echo "Submitting evaluation for: $exp"
    JOB_ID=$(sbatch --parsable run_eval_checkpoints.slurm "$exp")
    echo "  Job ID: $JOB_ID"
done

echo ""
echo "All evaluation jobs submitted!"
echo "Track progress with: squeue -u \$USER"
echo ""
echo "Results will be saved in each experiment directory:"
for exp in "${EXPERIMENTS[@]}"; do
    echo "  $exp/checkpoint_evaluations.json"
done