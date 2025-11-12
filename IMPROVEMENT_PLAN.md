# Improved TrOCR Training to Fix CER and Prediction Quality

## Problem Analysis

The current TrOCR model has severe issues:
- **High CER**: 0.84 (84% character error rate) - should be <0.1 for good performance
- **Strange predictions**: Getting "ข29 62 0114" instead of "กง8989" (Thai license plates)
- **Zero exact matches**: No complete plate predictions are correct
- **Province prediction**: 100% error rate on province classification

## Root Causes Identified

1. **Tokenizer/Embedding Mismatch**
   - Original model: 25,005 tokens  
   - With province tokens: 25,082 tokens
   - New embeddings randomly initialized, corrupting pre-trained representations

2. **Province Prediction Complexity**
   - Adding 77 province tokens makes training much harder
   - Model confused between plate text and province classification
   - Training target format: "กง8989 <prov> <TH-70>" is complex

3. **Suboptimal Training Parameters**
   - Learning rate too high (2e-5) for fine-tuning
   - Not enough epochs (5) for convergence
   - Batch sizes may be too large for stability

## Improvement Strategy

### Three Parallel Experiments

1. **Plate-Only Training** (`run_plate_only_improved.slurm`)
   - Remove province prediction entirely
   - Focus only on Thai license plate text recognition
   - Simpler training target: just "กง8989"
   - Expected: Significant CER improvement

2. **High-Quality Training** (`run_plate_high_quality.slurm`) 
   - Optimal parameters for lowest possible CER
   - Lower learning rate (5e-6) for stability
   - More epochs (15) for complete convergence
   - Medium augmentation for robustness
   - Multiple beam search evaluations

3. **Kkatiz Optimized** (`run_kkatiz_optimized.slurm`)
   - Use `kkatiz/thai-trocr-thaigov-v2` model
   - Pre-trained specifically on Thai government plates
   - Should perform better than openthaigpt baseline

### Key Improvements Applied

**Training Parameters:**
- Lower learning rates: 1e-5 to 5e-6 (vs 2e-5)
- More epochs: 8-15 (vs 5)
- Smaller batch sizes: 2-4 (vs 8) for stability
- Higher gradient accumulation: 2-4 steps
- Better warmup ratio: 0.1-0.15

**Model Configuration:**
- Remove province prediction complexity
- Shorter generation length: 16-20 tokens
- Focus on eval_cer metric for optimization
- Early stopping for best performance

**Data Handling:**
- Light/medium augmentation only
- Better evaluation strategy (epoch-based)
- Multiple beam search settings

## Expected Results

**Target Performance:**
- CER: <0.2 (vs current 0.84)
- Exact match: >70% (vs current 0%)
- Proper Thai predictions: "กง8989", "5กณ2033" etc.
- Training time: 2-4 hours per experiment

**Success Indicators:**
- Predictions look like real Thai plates
- No more strange patterns like "ข29 0110 10"
- Significant CER reduction
- High exact match percentage

## Monitoring

Jobs submitted:
- Job 30473: plate_only_improved
- Job 30474: plate_high_quality  
- Job 30475: kkatiz_optimized

Results will be saved in:
- `outputs/plate_only_improved/`
- `outputs/plate_high_quality/`
- `outputs/kkatiz_optimized/`

Monitor with: `squeue -u $USER`

## Next Steps

1. Wait for training completion (2-6 hours)
2. Compare evaluation results from all three experiments
3. Select best performing model
4. If needed, add province prediction back with simplified approach
5. Deploy best model for production use