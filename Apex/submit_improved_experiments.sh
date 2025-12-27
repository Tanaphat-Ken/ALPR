#!/bin/bash

echo "Submitting improved TrOCR experiments to fix CER and prediction quality..."

# Submit plate-only training (no province complexity)
echo "1. Submitting plate-only improved training..."
PLATE_ONLY_JOB=$(sbatch --parsable run_plate_only_improved.slurm)
echo "   Job ID: $PLATE_ONLY_JOB"

# Submit high-quality training with optimal settings
echo "2. Submitting high-quality training..."
HIGH_QUALITY_JOB=$(sbatch --parsable run_plate_high_quality.slurm)
echo "   Job ID: $HIGH_QUALITY_JOB"

# Submit kkatiz optimized training
echo "3. Submitting kkatiz optimized training..."
KKATIZ_JOB=$(sbatch --parsable run_kkatiz_optimized.slurm)
echo "   Job ID: $KKATIZ_JOB"

echo ""
echo "All jobs submitted successfully!"
echo "Track progress with: squeue -u \$USER"
echo ""
echo "Expected improvements:"
echo "- Remove province prediction complexity (reduces CER)"
echo "- Lower learning rate for better convergence"
echo "- More epochs for proper training"
echo "- Better augmentation settings"
echo "- Optimized batch sizes and gradient accumulation"
echo ""
echo "Monitor results in:"
echo "- outputs/plate_only_improved/"
echo "- outputs/plate_high_quality/"
echo "- outputs/kkatiz_optimized/"