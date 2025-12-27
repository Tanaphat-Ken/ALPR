#!/bin/bash
# Master script to submit all experiments in parallel

echo "Submitting all TrOCR experiments in parallel..."
echo "Each experiment will run on its own A100 GPU"
echo "=========================================="

# Submit all experiments
echo "Submitting exp1_v1_clean..."
JOB1=$(sbatch run_exp1_v1_clean.slurm | awk '{print $4}')
echo "Job ID: $JOB1"

echo "Submitting exp2_v1_aug..."
JOB2=$(sbatch run_exp2_v1_aug.slurm | awk '{print $4}')
echo "Job ID: $JOB2"

echo "Submitting exp3_kkatiz_clean..."
JOB3=$(sbatch run_exp3_kkatiz_clean.slurm | awk '{print $4}')
echo "Job ID: $JOB3"

echo "Submitting exp4_kkatiz_aug..."
JOB4=$(sbatch run_exp4_kkatiz_aug.slurm | awk '{print $4}')
echo "Job ID: $JOB4"

echo "Submitting exp5_openthaigpt_aug..."
JOB5=$(sbatch run_exp5_openthaigpt_aug.slurm | awk '{print $4}')
echo "Job ID: $JOB5"

echo ""
echo "All experiments submitted! Job IDs:"
echo "exp1_v1_clean: $JOB1"
echo "exp2_v1_aug: $JOB2"
echo "exp3_kkatiz_clean: $JOB3"
echo "exp4_kkatiz_aug: $JOB4"
echo "exp5_openthaigpt_aug: $JOB5"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check logs with: tail -f slurm_exp*_\$JOBID.out"
echo ""
echo "To cancel all experiments:"
echo "scancel $JOB1 $JOB2 $JOB3 $JOB4 $JOB5"