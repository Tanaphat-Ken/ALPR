"""Evaluate TrOCR checkpoints on license-plate datasets.

The script mirrors the training pipeline and reports:
- Character Error Rate (CER)
- Exact-match plate accuracy
- Average latency per image & throughput

It supports reading either JSONL manifests produced by ``train_trocr.py`` or
recomputing the stratified CSV splits on the fly (same seed/ratios).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from train.data_utils import LicensePlateRecord, load_plate_crop, load_records, stratified_split
from train.train_trocr import character_error_rate, load_model_and_processor, build_label_with_province, split_prediction_and_province

logger = logging.getLogger("eval_trocr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TrOCR checkpoints on plate crops")
    parser.add_argument("--csv", type=str, help="Path to dataset CSV (tb_match_...)")
    parser.add_argument("--data-root", type=str, action="append", help="Root directory containing the /210/... hierarchy.")
    parser.add_argument("--manifest", type=str, help="Optional JSONL manifest produced by train_trocr.py")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate when using CSV.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model-id", type=str, default="kkatiz/thai-trocr-thaigov-v2", help="Base model identifier (used when --model-path is a .pth state_dict).")
    parser.add_argument("--model-path", type=str, required=True, help="Checkpoint directory or .pth state_dict to evaluate.")
    parser.add_argument("--generation-max-length", type=int, default=32)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-results", type=str, help="Optional path to dump per-sample JSON results.")
    parser.add_argument("--normalize-text", action="store_true", help="Strip whitespace before comparison.")
    parser.add_argument("--use-lora", action="store_true", help="Load LoRA adapters from model-path directory if present.")
    parser.add_argument("--predict-province", action="store_true", help="Evaluate province prediction alongside plate text.")
    parser.add_argument("--province-format", type=str, default="code", choices=["code", "name"], help="Province format used during training.")
    return parser.parse_args()


def load_manifest(path: Path) -> List[LicensePlateRecord]:
    records: List[LicensePlateRecord] = []
    with path.open(encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
            payload = json.loads(line)
            records.append(
                LicensePlateRecord(
                    index=idx,
                    plate_text=payload["text"],
                    plate_image_path=Path(payload["image_path"]).resolve(),
                    raw_image_path=None,
                    cameras_plate_no=payload.get("cameras_plate_no"),
                    province_code=payload.get("province_code"),
                    province_description=payload.get("province_description"),
                    is_validate=True,
                    car_bbox=None,
                    plate_bbox=None,
                    character_bboxes=None,
                    metadata=payload.get("metadata", {}),
                )
            )
    logger.info("Loaded %d samples from manifest %s", len(records), path)
    return records


def get_records(args: argparse.Namespace) -> List[LicensePlateRecord]:
    if args.manifest:
        return load_manifest(Path(args.manifest))

    if not args.csv or not args.data_root:
        raise ValueError("When --manifest is not provided, --csv and --data-root are required.")

    if not math.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0, rel_tol=1e-3):
        raise ValueError("train/val/test ratios must sum to 1.0")

    records = load_records(
        args.csv,
        data_roots=args.data_root,
        require_validate=True,
        drop_missing_images=True,
    )
    splits = stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    selected = splits[args.split]
    logger.info("Selected %d samples from %s split", len(selected), args.split)
    return selected


def prepare_model(args: argparse.Namespace):
    ns = SimpleNamespace(
        model_id=args.model_id,
        model_path=args.model_path,
        generation_max_length=args.generation_max_length,
        freeze_encoder=False,
        decoder_only=False,
        use_lora=args.use_lora,
        lora_target_modules="auto",
        lora_scope="decoder",
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        predict_province=getattr(args, "predict_province", False),
        province_format=getattr(args, "province_format", "code"),
    )
    model, processor = load_model_and_processor(ns)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()
    return model, processor, device


def batched(iterable: Sequence[LicensePlateRecord], batch_size: int) -> Iterable[Sequence[LicensePlateRecord]]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i : i + batch_size]


def evaluate(args: argparse.Namespace) -> Dict[str, float]:
    records = get_records(args)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    model, processor, device = prepare_model(args)

    all_results = []
    cer_scores = []
    exact_matches = 0
    latencies = []
    total_images = 0
    total_time = 0.0
    
    # Additional metrics for province prediction
    if args.predict_province:
        plate_cer_scores = []
        province_cer_scores = []
        plate_exact_matches = 0
        province_exact_matches = 0

    pad_token_id = processor.tokenizer.pad_token_id

    for batch in batched(records, args.batch_size):
        try:
            images = [load_plate_crop(rec) for rec in batch]
            inputs = processor(images=images, return_tensors="pt").to(device)

            torch.cuda.synchronize(device) if device.type == "cuda" else None
            start = time.perf_counter()
            
            with torch.no_grad():  # Ensure no gradients to save memory
                output_ids = model.generate(
                    **inputs,
                    num_beams=args.num_beams,
                    max_length=args.generation_max_length,
                    early_stopping=True if args.num_beams > 1 else False,
                    no_repeat_ngram_size=2 if args.num_beams > 1 else None,
                    pad_token_id=pad_token_id,
                )
            
            torch.cuda.synchronize(device) if device.type == "cuda" else None
            elapsed = time.perf_counter() - start

            total_time += elapsed
            batch_size_actual = len(batch)
            latencies.append(elapsed / batch_size_actual)
            total_images += batch_size_actual

            preds = processor.batch_decode(output_ids, skip_special_tokens=True)
            
            # Clear GPU memory after each batch
            del inputs, output_ids
            if device.type == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Skip this batch and continue
            continue

        for rec, pred in zip(batch, preds):
            if args.predict_province:
                # Build ground truth with province
                label = build_label_with_province(
                    rec.plate_text, 
                    rec.province_code, 
                    args.province_format
                )
                candidate = pred.strip() if args.normalize_text else pred
                
                # Split predictions and labels into plate and province parts
                pred_plate, pred_province = split_prediction_and_province(candidate)
                label_plate, label_province = split_prediction_and_province(label)
                
                # Calculate separate metrics
                plate_cer = character_error_rate(pred_plate, label_plate)
                province_cer = character_error_rate(pred_province, label_province)
                combined_cer = character_error_rate(candidate, label)
                
                plate_cer_scores.append(plate_cer)
                province_cer_scores.append(province_cer)
                cer_scores.append(combined_cer)
                
                if pred_plate.strip() == label_plate.strip():
                    plate_exact_matches += 1
                if pred_province.strip() == label_province.strip():
                    province_exact_matches += 1
                if candidate.strip() == label.strip():
                    exact_matches += 1
                
                all_results.append({
                    "index": rec.index,
                    "image_path": str(rec.plate_image_path),
                    "prediction": candidate,
                    "ground_truth": label,
                    "pred_plate": pred_plate,
                    "pred_province": pred_province,
                    "label_plate": label_plate,
                    "label_province": label_province,
                    "cer": combined_cer,
                    "plate_cer": plate_cer,
                    "province_cer": province_cer,
                    "province_code": rec.province_code,
                })
            else:
                # Original logic for plate text only
                label = rec.plate_text.strip() if args.normalize_text else rec.plate_text
                candidate = pred.strip() if args.normalize_text else pred
                cer = character_error_rate(candidate, label)
                cer_scores.append(cer)
                if candidate == label:
                    exact_matches += 1
                all_results.append({
                    "index": rec.index,
                    "image_path": str(rec.plate_image_path),
                    "prediction": candidate,
                    "ground_truth": label,
                    "cer": cer,
                    "province_code": rec.province_code,
                })

    # Build metrics
    metrics = {
        "samples": total_images,
        "cer": float(mean(cer_scores) if cer_scores else 0.0),
        "exact_match": float(exact_matches / total_images if total_images else 0.0),
        "latency_ms": float(mean(latencies) * 1000 if latencies else 0.0),
        "throughput_ips": float(total_images / total_time if total_time else 0.0),
    }
    
    if args.predict_province:
        metrics.update({
            "plate_cer": float(mean(plate_cer_scores) if plate_cer_scores else 0.0),
            "plate_exact_match": float(plate_exact_matches / total_images if total_images else 0.0),
            "province_cer": float(mean(province_cer_scores) if province_cer_scores else 0.0),
            "province_exact_match": float(province_exact_matches / total_images if total_images else 0.0),
        })

    logger.info("Evaluation complete: %s", metrics)

    if args.save_results:
        out_path = Path(args.save_results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            json.dump({"metrics": metrics, "results": all_results}, fp, ensure_ascii=False, indent=2)
        logger.info("Saved detailed results to %s", out_path)

    return metrics


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    metrics = evaluate(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
