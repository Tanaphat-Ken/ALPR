#!/bin/bash
# Script to submit just the most promising experiments for testing

echo "Submitting 2 best experiments for quick testing..."
echo "================================================"

# Submit the two fastest/most promising experiments
echo "Submitting exp3_kkatiz_clean (fastest, clean data)..."
JOB3=$(sbatch run_exp3_kkatiz_clean.slurm | awk '{print $4}')
echo "Job ID: $JOB3"

echo "Submitting exp5_openthaigpt_aug (good balance)..."
JOB5=$(sbatch run_exp5_openthaigpt_aug.slurm | awk '{print $4}')
echo "Job ID: $JOB5"

echo ""
echo "Test experiments submitted! Job IDs:"
echo "exp3_kkatiz_clean: $JOB3"
echo "exp5_openthaigpt_aug: $JOB5"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs with: tail -f slurm_exp*_\$JOBID.out"
echo ""
echo "To cancel test experiments:"
echo "scancel $JOB3 $JOB5"