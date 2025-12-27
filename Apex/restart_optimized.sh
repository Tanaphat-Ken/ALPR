#!/bin/bash

# Cancel current slow jobs
echo "Canceling current jobs..."
scancel -u $USER

# Wait a moment
sleep 2

# Submit optimized job
echo "Submitting optimized job..."
sbatch run_experiments_test.slurm

echo "New job submitted with optimized parameters:"
echo "- Batch size: 32 → 16 (test script)"
echo "- Workers: 8 → 2 (reduced I/O contention)"
echo "- Epochs: 5 → 2 (faster testing)"
echo "- Beams: 5 → 3 (faster inference)"

echo ""
echo "Check job status with: squeue -u \$USER"
echo "Monitor with: tail -f slurm_test_output_*.out"