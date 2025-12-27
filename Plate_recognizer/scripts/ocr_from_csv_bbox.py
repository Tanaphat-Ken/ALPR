"""
OCR from CSV plate_bbox: crops full images by CSV bounding boxes and runs TrOCR.

Usage (PowerShell):
  python scripts/ocr_from_csv_bbox.py --csv data/tb_match_data_20240705_10581-11080.csv \
      --filter-substr "/210/2024-07-05/14/Lane1/" --limit 100 --metrics

Notes:
- CSV columns expected: image_name_gray (or image_name), plate_bbox, plate (gt)
- plate_bbox is expected in format like "[[x1, y1, x2, y2]]" (absolute coords on the full image)
- If your CSV uses a different shape, adjust the parsing in parse_bbox()
"""
from __future__ import annotations
import os
import sys
import csv
import ast
import argparse
from typing import List, Optional, Tuple
from PIL import Image

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

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


def parse_bbox(s: str) -> Optional[Tuple[int,int,int,int]]:
    if not s:
        return None
    try:
        val = ast.literal_eval(s)
        # Expect [[x1, y1, x2, y2]] or [x1, y1, x2, y2]
        if isinstance(val, list) and len(val) == 1 and isinstance(val[0], (list, tuple)):
            x1, y1, x2, y2 = val[0]
        elif isinstance(val, (list, tuple)) and len(val) == 4:
            x1, y1, x2, y2 = val
        else:
            return None
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None


def load_rows(csv_path: str, filter_substr: Optional[str], limit: Optional[int], data_roots: Optional[List[str]] = None):
    rows = []
    data_roots = data_roots or []

    def resolve_path(img_rel: str) -> Optional[str]:
        rel = img_rel.lstrip("/\\")
        for root in data_roots:
            cand = os.path.join(root, rel)
            if os.path.exists(cand):
                return cand
        cand1 = os.path.join(ROOT_DIR, "data", rel)
        if os.path.exists(cand1):
            return cand1
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
            bbox = parse_bbox(r.get('plate_bbox', ''))
            if not bbox:
                continue
            gt = r.get('plate')
            img_path = resolve_path(img_rel)
            if not img_path:
                if len(rows) < 3:
                    print(f"[warn] cannot resolve path for: {img_rel}")
                continue
            rows.append((img_path, bbox, gt))
            if limit and len(rows) >= limit:
                break
    return rows


def crop(img: Image.Image, bbox: Tuple[int,int,int,int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = img.size
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def run(rows, metrics: bool):
    reader = CharacterReader()
    num = len(rows)
    acc_cnt = 0
    cer_sum = 0.0

    for i, (p, bbox, gt) in enumerate(rows, 1):
        try:
            img = Image.open(p).convert('RGB')
        except Exception as e:
            print(f"[{i}/{num}] ERROR opening {p}: {e}")
            continue
        plate_img = crop(img, bbox)
        pred_list = reader.predict([plate_img])
        pred = pred_list[0] if pred_list else ''
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


def parse_args():
    ap = argparse.ArgumentParser(description="OCR from CSV plate bboxes")
    ap.add_argument("--csv", type=str, required=True, help="CSV path with image_name_gray, plate_bbox, plate")
    ap.add_argument("--filter-substr", type=str, default=None, help="Substring filter for image path in CSV")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    ap.add_argument("--metrics", action='store_true', help="Compute accuracy and CER")
    ap.add_argument("--data-root", type=str, default=None, help="Root folder to resolve CSV image paths (e.g., data\\210-20250930T155802Z-1-001)")

    # Optional override for TrOCR weights
    ap.add_argument("--weights-dir", type=str, default=None, help="Path to TrOCR save_pretrained directory")
    ap.add_argument("--weights-pth", type=str, default=None, help="Path to TrOCR .pth state_dict file")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.weights_dir and os.path.isdir(args.weights_dir):
        os.environ['CHARACTOR_READER_WEIGHT'] = args.weights_dir
        configs.CHARACTOR_READER_WEIGHT = args.weights_dir
    elif args.weights_pth and os.path.isfile(args.weights_pth):
        os.environ['CHARACTOR_READER_WEIGHT'] = args.weights_pth
        configs.CHARACTOR_READER_WEIGHT = args.weights_pth

    data_roots: List[str] = []
    if args.data_root:
        data_roots.append(args.data_root)
    data_roots.append(os.path.join(ROOT_DIR, "data", "210-20250930T155802Z-1-001"))
    data_roots.append(os.path.join(ROOT_DIR, "data"))

    rows = load_rows(args.csv, args.filter_substr, args.limit, data_roots=data_roots)
    if not rows:
        print("No rows matched. Check --filter-substr or CSV columns.")
        return
    run(rows, metrics=args.metrics)


if __name__ == "__main__":
    main()
