#!/bin/bash
# Script to diagnose and fix slow training performance

echo "=== Training Performance Diagnostics ==="
echo "======================================="

# Check current jobs
echo "Current SLURM jobs:"
squeue -u $USER

echo ""
echo "=== Performance Analysis ==="

# Option 1: Cancel slow jobs and run minimal test
echo "1. Cancel slow jobs and run minimal test (recommended first step):"
echo "   scancel 30459 30460 30461"  
echo "   sbatch run_minimal_test.slurm"

echo ""
# Option 2: Run optimized version
echo "2. Run ultra-optimized exp3 (with SLURM optimizations):"
echo "   sbatch run_exp3_slurm_optimized.slurm"

echo ""
# Option 3: Run fast version
echo "3. Run fast version (reduced batch size, epochs):"
echo "   sbatch run_exp3_fast.slurm"

echo ""
echo "=== Root Cause Analysis ==="
echo "Current issues:"
echo "- Effective batch size ~8 instead of 32 (memory/gradient accumulation issue)"
echo "- 50-60s per step (should be 5-8s)"
echo "- Possible I/O bottleneck or GPU underutilization"

echo ""
echo "=== Optimizations Applied ==="
echo "1. Reduced batch size: 32 → 16/24"
echo "2. Added gradient accumulation for effective larger batches"
echo "3. Disabled data workers (num_workers=0) to avoid I/O contention" 
echo "4. Enabled exclusive GPU access"
echo "5. Set GPU to performance mode"
echo "6. Reduced epochs for faster testing"

echo ""
echo "Recommended action: Run minimal test first to validate performance!"