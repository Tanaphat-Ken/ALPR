"""
Run the full ALPR pipeline (car -> plate -> OCR) over a directory of images or a CSV file.

Usage examples (PowerShell):
  # Run over a folder of images
  python scripts/run_full_pipeline.py --images data/210/2024-07-05/14/Lane1 --limit 50

  # Run using CSV listing (requires columns: image_name_gray, plate)
  python scripts/run_full_pipeline.py --csv data/tb_match_data_20240705_10581-11080.csv --filter-substr "/210/2024-07-05/14/Lane1/" --limit 100

Options:
  --weights-dir or --weights-pth can point to your TrOCR weights; otherwise configs.CHARACTOR_READER_WEIGHT is used.
  Use --metrics to compute accuracy & CER when ground truth is available.
"""
from __future__ import annotations
import os
import sys
import csv
import argparse
from typing import List, Optional, Tuple
from PIL import Image

# Ensure we can import project modules when running from repo root
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from models.image_processor import ImageProcessor
from models.localizers import CharacterReader
from constants import configs

# Optional Levenshtein for CER
try:
    import Levenshtein  # type: ignore
except Exception:
    Levenshtein = None


def char_error_rate(pred: str, gt: str) -> float:
    if gt is None:
        return 1.0
    gt = gt or ""
    pred = pred or ""
    if len(gt) == 0 and len(pred) == 0:
        return 0.0
    if Levenshtein:
        dist = Levenshtein.distance(pred, gt)
    else:
        # simple DP edit distance
        m, n = len(pred), len(gt)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                cost = 0 if pred[i - 1] == gt[j - 1] else 1
                dp[j] = min(
                    dp[j] + 1,      # deletion
                    dp[j - 1] + 1,  # insertion
                    prev + cost     # substitution
                )
                prev = temp
        dist = dp[n]
    return dist / max(1, len(gt))


def load_images_from_dir(dir_path: str, limit: Optional[int]) -> List[str]:
    imgs = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                imgs.append(os.path.join(root, f))
    imgs.sort()
    if limit:
        imgs = imgs[:limit]
    return imgs


def load_from_csv(csv_path: str, filter_substr: Optional[str], limit: Optional[int], data_roots: Optional[List[str]] = None) -> List[Tuple[str, Optional[str]]]:
    rows = []
    data_roots = data_roots or []

    def resolve_path(img_rel: str) -> Optional[str]:
        # Normalize leading slash/backslash: "/210/..." -> "210/..."
        rel = img_rel.lstrip("/\\")

        # Try user-provided data roots first
        for root in data_roots:
            cand = os.path.join(root, rel)
            if os.path.exists(cand):
                return cand

        # Try common repo defaults
        cand1 = os.path.join(ROOT_DIR, "data", rel)
        if os.path.exists(cand1):
            return cand1

        # As a last resort, try joining to ROOT_DIR directly
        cand2 = os.path.join(ROOT_DIR, img_rel)
        if os.path.exists(cand2):
            return cand2

        return None

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_rel = r.get('image_name_gray') or r.get('image_name') or ''
            if not img_rel:
                continue
            if filter_substr and filter_substr not in img_rel:
                continue
            gt = r.get('plate')
            img_path = resolve_path(img_rel)
            if not img_path:
                # Print first few warnings to help debugging
                if len(rows) < 3:
                    print(f"[warn] cannot resolve path for: {img_rel}")
                continue
            rows.append((img_path, gt))
            if limit and len(rows) >= limit:
                break
    return rows


def run_over_images(img_paths: List[str], metrics: bool, use_craft: bool) -> None:
    ip = ImageProcessor(use_craft=use_craft)
    num = len(img_paths)
    acc_cnt = 0
    cer_sum = 0.0

    for i, p in enumerate(img_paths, 1):
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f"[{i}/{num}] ERROR opening {p}: {e}")
            continue
        out = ip.read(img)
        pred = out.get('full_plate') or ''
        print(f"[{i}/{num}] {os.path.basename(p)} -> {pred}")

    if metrics:
        print("--metrics requested, but no ground truths were provided with --images. Use --csv to compute metrics.")


def run_over_csv(rows: List[Tuple[str, Optional[str]]], metrics: bool, use_craft: bool) -> None:
    ip = ImageProcessor(use_craft=use_craft)
    num = len(rows)
    acc_cnt = 0
    cer_sum = 0.0

    for i, (p, gt) in enumerate(rows, 1):
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f"[{i}/{num}] ERROR opening {p}: {e}")
            continue
        out = ip.read(img)
        pred = out.get('full_plate') or ''
        print(f"[{i}/{num}] {os.path.basename(p)} -> {pred} | gt={gt}")
        if metrics and gt is not None:
            if pred == gt:
                acc_cnt += 1
            cer_sum += char_error_rate(pred, gt)

    if metrics:
        total = len(rows)
        acc = acc_cnt / total if total else 0.0
        cer = cer_sum / total if total else 0.0
        print(f"\nMetrics on {total} samples:")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  CER:      {cer:.4f}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run full ALPR pipeline")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--images", type=str, help="Directory of images to run")
    g.add_argument("--csv", type=str, help="CSV file with 'image_name_gray' and optional 'plate'")

    ap.add_argument("--filter-substr", type=str, default=None, help="Substring to filter CSV rows (e.g. /210/2024-07-05/14/Lane1/)")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    ap.add_argument("--metrics", action='store_true', help="Compute accuracy and CER when ground truth available")
    ap.add_argument("--use-craft", action='store_true', help="Use CRAFT to detect text regions inside the plate crop before OCR")
    ap.add_argument("--data-root", type=str, default=None, help="Root folder to resolve CSV image paths (e.g., data\\210-20250930T155802Z-1-001)")

    # OCR weights override (optional)
    ap.add_argument("--weights-dir", type=str, default=None, help="Path to TrOCR save_pretrained directory")
    ap.add_argument("--weights-pth", type=str, default=None, help="Path to TrOCR .pth state_dict file")
    return ap.parse_args()


def main():
    args = parse_args()

    # If user wants to override CharacterReader weights globally, we can set env for configs
    if args.weights_dir and os.path.isdir(args.weights_dir):
        os.environ['CHARACTOR_READER_WEIGHT'] = args.weights_dir
        configs.CHARACTOR_READER_WEIGHT = args.weights_dir
    elif args.weights_pth and os.path.isfile(args.weights_pth):
        os.environ['CHARACTOR_READER_WEIGHT'] = args.weights_pth
        configs.CHARACTOR_READER_WEIGHT = args.weights_pth

    if args.images:
        img_paths = load_images_from_dir(args.images, args.limit)
        run_over_images(img_paths, metrics=args.metrics, use_craft=args.use_craft)
    else:
        # Build candidate data roots: user-provided first, then common defaults
        data_roots: List[str] = []
        if args.data_root:
            data_roots.append(args.data_root)
        # Known archive folder seen in this repo
        data_roots.append(os.path.join(ROOT_DIR, "data", "210-20250930T155802Z-1-001"))
        data_roots.append(os.path.join(ROOT_DIR, "data"))

        rows = load_from_csv(args.csv, args.filter_substr, args.limit, data_roots=data_roots)
        if not rows:
            print("No rows resolved. Try providing --data-root to the folder that contains the '210/...' structure.")
        run_over_csv(rows, metrics=args.metrics, use_craft=args.use_craft)


if __name__ == "__main__":
    main()
