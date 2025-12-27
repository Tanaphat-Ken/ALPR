# TrOCR Fine-tuning PoC Toolkit

This directory packages everything needed to run the PoC described in your
plan: dataset preparation, configurable fine-tuning for two Thai TrOCR
checkpoints, evaluation utilities, and recommended experiment recipes.

## 1. Environment setup

Use the project virtual environment then install the extra libraries needed for
Hugging Face training:

```powershell
cd /d D:\CodingD\ALPR-V2\alpr_service\plate_recognizer
.\.venv\Scripts\activate
pip install "transformers[torch]" datasets accelerate peft sentencepiece "python-Levenshtein>=0.22" numpy
```

(If you plan to log to Weights & Biases add `wandb`.)

## 2. Dataset preparation

The CSV `data/tb_match_data_20240705_10581-11080.csv` already contains plate
crops (`image_name_gray`), labels, province codes, and validation flags.

`train/data_utils.py` exposes helpers to load and split the dataset.

Example – build manifests for later reuse:

```powershell
.\.venv\Scripts\python.exe - <<'PY'
from train import data_utils
records = data_utils.load_records(
    "data/tb_match_data_20240705_10581-11080.csv",
    data_roots=["data/210-20250930T155802Z-1-001"],
)
splits = data_utils.stratified_split(records, seed=42)
data_utils.export_manifest(splits, "train/manifests", include_metadata=True)
PY
```

This creates `train/manifests/train.jsonl`, `val.jsonl`, and `test.jsonl` which
feed directly into the evaluation script.

## 3. Training (`train_trocr.py`)

The training script supports both baseline checkpoints:

- `openthaigpt/thai-trocr`
- `kkatiz/thai-trocr-thaigov-v2`

and warm-starting from your internal V1 `.pth` weights. Key features:

- Stratified train/val/test splits and optional manifest export
- Pillow-based augmentations with presets (`--augment none|light|medium|heavy`)
- Mixed precision (`--fp16` or `--bf16`)
- LoRA fine-tuning (`--use-lora`, configurable rank/alpha/targets)
- Encoder freezing (`--freeze-encoder` / `--decoder-only`)
- Automatic validation & test evaluation with metric logging

### Minimal runs

Baseline (clean data, no LoRA):

```powershell
.\.venv\Scripts\python.exe train\train_trocr.py \
  --csv data\tb_match_data_20240705_10581-11080.csv \
  --data-root data\210-20250930T155802Z-1-001 \
  --model-id kkatiz/thai-trocr-thaigov-v2 \
  --augment none \
  --output-dir outputs\kkatiz_clean
```

Augmented + full fine-tuning (recommended):

```powershell
.\.venv\Scripts\python.exe train\train_trocr.py \
  --csv data\tb_match_data_20240705_10581-11080.csv \
  --data-root data\210-20250930T155802Z-1-001 \
  --model-id kkatiz/thai-trocr-thaigov-v2 \
  --augment heavy \
  --fp16 \
  --predict-province --province-format code \
  --output-dir outputs\kkatiz_aug \
  --report-to wandb  # optional
```

Continuation from V1 weights with province prediction:

```powershell
.\.venv\Scripts\python.exe train\train_trocr.py \
  --csv data\tb_match_data_20240705_10581-11080.csv \
  --data-root data\210-20250930T155802Z-1-001 \
  --model-id openthaigpt/thai-trocr \
  --model-path models\weights\charactor_reader.pth \
  --augment medium \
  --freeze-encoder \
  --predict-province --province-format code \
  --output-dir outputs\v1_medium
```

`train_trocr.py --help` lists every option (batch sizes, schedulers, manifests,
resume, etc.).

## 4. Evaluation (`eval_trocr.py`)

Use this script to benchmark CER, exact-match accuracy, and latency. You can
point it either at the CSV (it will regenerate splits with the same seed) or at
the manifests saved earlier.

Example using the generated manifests:

```powershell
.\.venv\Scripts\python.exe train\eval_trocr.py \
  --manifest train\manifests\test.jsonl \
  --model-path outputs\kkatiz_aug \
  --model-id kkatiz/thai-trocr-thaigov-v2 \
  --num-beams 5 --batch-size 4 --normalize-text \
  --predict-province --province-format code \
  --save-results outputs\kkatiz_aug\test_results.json
```

Example directly from CSV (recomputes split with `seed=42`):

```powershell
.\.venv\Scripts\python.exe train\eval_trocr.py \
  --csv data\tb_match_data_20240705_10581-11080.csv \
  --data-root data\210-20250930T155802Z-1-001 \
  --model-path outputs\v1_medium \
  --model-id openthaigpt/thai-trocr \
  --split test --num-beams 5 --batch-size 2 \
  --predict-province --province-format code
```

The script prints metrics to STDOUT and (optionally) writes a detailed JSON file
with per-sample predictions.

## 5. Recommended experiment grid

| #   | Model init  | Data      | Augment | Special        | Notes                                 |
| --- | ----------- | --------- | ------- | -------------- | ------------------------------------- |
| 1   | V1 (.pth)   | Clean     | none    | full fine-tune | Baseline continuation with provinces  |
| 2   | V1 (.pth)   | Augmented | heavy   | full fine-tune | Measures impact of augment only       |
| 3   | kkatiz      | Clean     | none    | full fine-tune | Direct baseline with province support |
| 4   | kkatiz      | Augmented | heavy   | full fine-tune | Primary candidate with provinces      |
| 5   | openthaigpt | Augmented | medium  | full fine-tune | Backup checkpoint with provinces      |

**Province Prediction:** All experiments include province prediction support with Thai province code format (`<TH-XX>`). The training pipeline automatically adds 77 province tokens and handles vocabulary expansion. Each model learns to predict both license plate text and province codes in the format: `platetext <prov> <TH-XX>`.

**Full Fine-tuning:** All experiments use full model fine-tuning instead of LoRA for better performance and stability. This approach trains all model parameters and provides more reliable results for the Thai license plate recognition task.

Track `cer`, `exact_match`, `plate_cer`, `plate_exact_match`, `province_cer`, `province_exact_match`, `latency_ms`, `throughput_ips` from `eval_trocr.py` and compare against baseline performance.

## 6. Tips & troubleshooting

- GPU memory tight? Add `--gradient-accumulation-steps 2` and reduce per-device
  batch size to 4.
- If augmentations make training unstable, switch to `--augment light`.
- To resume a run: `--resume-from-checkpoint outputs\kkatiz_aug\checkpoint-XXXX`.
- For province prediction evaluation, always include `--predict-province --province-format code` flags.
- Keep raw checkpoints out of Git: store under `outputs/` (already ignored) and
  push selected weights to an artifact store or release tag instead.
- Province tokens are automatically added during training - no manual preprocessing needed.

With these utilities you can reproduce the five requested PoC tracks across both reference models with full province prediction support and produce a concise report for stakeholders.

## 7. Automating the full grid

Use `train/run_experiments.py` to execute the five scenarios above in one go.
It will launch `train_trocr.py` followed by `eval_trocr.py` for each experiment,
dropping artefacts under a dedicated subfolder in `outputs/grid/`.

**Important:** The system now uses full fine-tuning instead of LoRA for better performance and stability. All experiments include automatic province prediction with Thai province codes.

Dry-run the pipeline first to inspect the generated commands:

```powershell
.\.venv\Scripts\python.exe train\run_experiments.py `
  --csv data\tb_match_data_20240705_10581-11080.csv `
  --data-root data\210-20250930T155802Z-1-001 `
  --dry-run
```

Run the full grid (training + evaluation) with standard settings:

```powershell
.\.venv\Scripts\python.exe train\run_experiments.py `
  --csv data\tb_match_data_20240705_10581-11080.csv `
  --data-root data\210-20250930T155802Z-1-001 `
  --output-root outputs\grid
```

For systems with limited GPU memory, use smaller batch sizes:

```powershell
.\.venv\Scripts\python.exe train\run_experiments.py `
  --csv data\tb_match_data_20240705_10581-11080.csv `
  --data-root data\210-20250930T155802Z-1-001 `
  --eval-batch-size 2 `
  --output-root outputs\grid
```

Alternative dataset example:

```powershell
.\.venv\Scripts\python.exe train\run_experiments.py `
  --csv data\8000\8000.csv `
  --data-root data\8000 `
  --output-root outputs\grid
```

Helpful switches:

- `--stage train` / `--stage eval` to run only one stage
- `--experiments exp1_v1_clean exp3_kkatiz_clean` to target specific experiments
- `--skip-existing` to avoid retraining when weights are already present
- `--max-train-samples 128 --eval-max-samples 64` for smoke tests
- `--eval-batch-size 2` for systems with limited GPU memory (default: 4)
- `--continue-on-error` to carry on even if one experiment fails
- `--num-workers 0` for Windows compatibility (required on Windows)

**Memory Management Tips:**

- Default evaluation batch size is now 4 (reduced from 8) for better memory efficiency
- The system includes automatic GPU memory cleanup between batches
- If you encounter memory issues, try `--eval-batch-size 2` or `--eval-batch-size 1`

Each evaluation writes a JSON report to `<output>/eval_test.json` with CER,
exact-match accuracy, plate/province-specific metrics, latency, throughput, and per-sample predictions
(when executed without `--dry-run`).

**Expected Output Structure:**

```
outputs/grid/
├── exp1_v1_clean/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── val_metrics.json
│   ├── test_metrics.json
│   └── eval_test.json
├── exp2_v1_aug/
├── exp3_kkatiz_clean/
├── exp4_kkatiz_aug/
└── exp5_openthaigpt_aug/
```

**Sample eval_test.json output:**

```json
{
  "metrics": {
    "samples": 145,
    "cer": 0.15,
    "exact_match": 0.75,
    "plate_cer": 0.12,
    "plate_exact_match": 0.8,
    "province_cer": 0.25,
    "province_exact_match": 0.65,
    "latency_ms": 85.3,
    "throughput_ips": 11.7
  },
  "results": [
    {
      "prediction": "2ขว1234 <prov> <TH-10>",
      "ground_truth": "2ขว1234 <prov> <TH-10>",
      "pred_plate": "2ขว1234",
      "pred_province": "TH-10",
      "label_plate": "2ขว1234",
      "label_province": "TH-10",
      "cer": 0.0,
      "plate_cer": 0.0,
      "province_cer": 0.0
    }
  ]
}
```
