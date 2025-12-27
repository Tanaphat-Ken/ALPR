import os, sys, csv, ast, time, argparse
from typing import Optional, Tuple, List, Dict
from PIL import Image, ImageOps

# Ensure package imports work whether run as "python scripts/..." or "-m scripts.ocr_testing"
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(FILE_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from models.localizers import CharacterReader  # uses configs.CHARACTOR_READER_WEIGHT by default


def resolve_path(rel: str, roots: List[str]) -> Optional[str]:
    rel = (rel or "").lstrip("/\\")
    for root in roots:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return None


def parse_rect(s: str) -> Optional[Tuple[int, int, int, int]]:
    try:
        arr = ast.literal_eval(s or "")
        if isinstance(arr, list) and arr and len(arr[0]) == 4:
            x1, y1, x2, y2 = map(int, map(round, arr[0]))
            return x1, y1, x2, y2
    except Exception:
        pass
    return None


def clamp_rect(r: Tuple[int, int, int, int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = r
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def levenshtein(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[lb]


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    t = s.strip()
    # remove spaces/underscores and unify common dashes
    t = t.replace(" ", "").replace("_", "").replace("–", "-").replace("—", "-")
    return t


def cer(ref: str, hyp: str, normalize: bool = True) -> float:
    ref = normalize_text(ref) if normalize else (ref or "").strip()
    hyp = normalize_text(hyp) if normalize else (hyp or "").strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / max(1, len(ref))


def preprocess(crop: Image.Image, height: int) -> Image.Image:
    """Preprocess crop for TrOCR: keep RGB and resize by height (preserve aspect)."""
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    h = height
    w = max(64, int(crop.width * (h / max(1, crop.height))))
    return crop.resize((w, h), Image.BILINEAR)


def run_generate(reader: CharacterReader, crops: List[Image.Image], num_beams: int, max_length: int) -> List[str]:
    """Generate text with optional beam search; always pass RGB images to the model/processor."""
    # Ensure RGB inputs to avoid "Unsupported number of image dimensions: 2"
    crops_rgb = [(c if getattr(c, "mode", "RGB") == "RGB" else c.convert("RGB")) for c in crops]

    model = getattr(reader, "model", None)
    processor = getattr(reader, "processor", None)
    if model is not None and processor is not None and num_beams and num_beams > 1:
        import torch
        model.eval()
        with torch.inference_mode():
            inputs = processor(images=crops_rgb, return_tensors="pt").to(model.device)
            ids = model.generate(**inputs, num_beams=num_beams, length_penalty=0.1, max_length=max_length)
            return [t.strip() for t in processor.batch_decode(ids, skip_special_tokens=True)]
    # fallback
    return [t.strip() for t in reader.predict(crops_rgb)]


def load_rows(csv_path: str, filter_substr: Optional[str], limit: Optional[int]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rel = r.get("image_name_gray") or r.get("image_name") or ""
            if not rel:
                continue
            if filter_substr and filter_substr not in rel:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser(description="Evaluate OCR using CSV plate_bbox (no detectors).")
    ap.add_argument("--csv", type=str, default=r"data\tb_match_data_20240705_10581-11080.csv")
    ap.add_argument("--filter-substr", type=str, default=r"/210/2024-07-05/14/Lane1/")
    ap.add_argument("--data-root", type=str, default=None, help=r'Root where CSV images live, e.g. "data\210-20250930T155802Z-1-001"')
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--num-beams", type=int, default=5, help="Beam search for generation; set 1 to disable.")
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--height", type=int, default=384, help="Preprocess resize height.")
    ap.add_argument("--no-preprocess", action="store_true", help="Disable grayscale+resize preprocessing.")
    ap.add_argument("--no-normalize", action="store_true", help="Disable text normalization before metrics.")
    ap.add_argument("--weights-dir", type=str, default=None, help="Optional: TrOCR directory from save_pretrained.")
    ap.add_argument("--weights-pth", type=str, default=None, help="Optional: TrOCR .pth state_dict file.")
    ap.add_argument("--show", type=int, default=8, help="Show N sample predictions.")
    ap.add_argument("--metrics", action="store_true", help="No-op: metrics are always printed.")
    args = ap.parse_args()

    # Candidate roots to resolve CSV paths
    candidate_roots = []
    if args.data_root:
        candidate_roots.append(args.data_root)
    candidate_roots.append(os.path.join(PROJ_ROOT, "data", "210-20250930T155802Z-1-001"))
    candidate_roots.append(os.path.join(PROJ_ROOT, "data"))

    rows = load_rows(args.csv, args.filter_substr, args.limit)
    print(f"Loaded {len(rows)} rows from {args.csv} filter={args.filter_substr!r}")

    # Init OCR
    reader = CharacterReader(weight_path=args.weights_dir or args.weights_pth)

    total = 0
    exact = 0
    sum_cer = 0.0
    shown = 0
    t0 = time.time()

    for i, r in enumerate(rows, 1):
        img_rel = r.get("image_name_gray") or r.get("image_name") or ""
        gt = (r.get("plate") or r.get("cameras_plateNo1") or "").strip()
        rect = parse_rect(r.get("plate_bbox") or "")

        img_path = resolve_path(img_rel, candidate_roots)
        if not (img_path and rect and os.path.exists(img_path)):
            print(f"[{i}] SKIP path/rect: {img_rel}")
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            x1, y1, x2, y2 = clamp_rect(rect, w, h)
            crop = img.crop((x1, y1, x2, y2))
            if not args.no_preprocess:
                crop = preprocess(crop, args.height)
            preds = run_generate(reader, [crop], num_beams=args.num_beams, max_length=args.max_length)
            pred = (preds[0] if preds else "").strip()
        except Exception as e:
            print(f"[{i}] OCR ERROR {img_rel}: {e}")
            continue

        total += 1
        gt_n = normalize_text(gt) if not args.no_normalize else gt
        pr_n = normalize_text(pred) if not args.no_normalize else pred
        if pr_n == gt_n:
            exact += 1
        sum_cer += cer(gt, pred, normalize=not args.no_normalize)

        if shown < args.show:
            print(f"[{i}] GT: {gt} | PRED: {pred} | file: {os.path.basename(img_path)}")
            shown += 1

        if i % 10 == 0 or i == len(rows):
            dt = time.time() - t0
            print(f"[{i}/{len(rows)}] acc={exact/max(1,total):.4f} cer={sum_cer/max(1,total):.4f} ips={total/max(1e-6,dt):.2f}")

    dt = time.time() - t0
    print("DONE:", {
        "samples": total,
        "exact_match_acc": round(exact/max(1,total), 4),
        "avg_cer": round(sum_cer/max(1,total), 4),
        "throughput_imgs_per_s": round(total/max(1e-6,dt), 2)
    })


if __name__ == "__main__":
    main()