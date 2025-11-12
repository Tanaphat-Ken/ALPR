#!/bin/bash
# Submit optimized experiments (avoiding the slow heavy augmentation)

echo "Submitting optimized experiments..."
echo "=================================="

# Cancel the slow heavy augmentation experiment
echo "Canceling slow exp2_v1_aug job..."
scancel 30458

# Submit the remaining optimized experiments  
echo "Submitting exp2_v1_medium (medium augmentation, faster)..."
JOB2=$(sbatch run_exp2_v1_medium.slurm | awk '{print $4}')
echo "Job ID: $JOB2"

echo "Submitting exp3_kkatiz_clean (optimized)..."
JOB3=$(sbatch run_exp3_kkatiz_clean.slurm | awk '{print $4}')
echo "Job ID: $JOB3"

echo "Submitting exp4_kkatiz_aug (heavy aug, but optimized batch size)..."
JOB4=$(sbatch run_exp4_kkatiz_aug.slurm | awk '{print $4}')
echo "Job ID: $JOB4"

echo "Submitting exp5_openthaigpt_aug (optimized)..."
JOB5=$(sbatch run_exp5_openthaigpt_aug.slurm | awk '{print $4}')
echo "Job ID: $JOB5"

echo ""
echo "Optimized experiments submitted!"
echo "exp1_v1_clean: 30457 (already running, keep it)"
echo "exp2_v1_medium: $JOB2"
echo "exp3_kkatiz_clean: $JOB3"  
echo "exp4_kkatiz_aug: $JOB4"
echo "exp5_openthaigpt_aug: $JOB5"
echo ""
echo "Monitor: squeue -u \$USER"
echo "Expected completion: ~2-3 hours each"