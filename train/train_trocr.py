"""Command-line training script for fine-tuning Thai TrOCR models.

Supports baseline huggingface checkpoints (e.g. ``openthaigpt/thai-trocr`` and
``kkatiz/thai-trocr-thaigov-v2``) as well as continuation from the in-house
"V1" model weights via ``--model-path``. The script integrates tightly with the
helpers in :mod:`train.data_utils` to build stratified splits from the provided
CSV manifest and to consume plate crops listed in ``image_name_gray``.

Key features
------------
- Deterministic stratified train/val/test splits from the CSV export
- Optional heavy augmentations that mimic motion blur, lighting, occlusion, etc.
- Mixed precision, gradient accumulation, and warm-up scheduling out of the box
- LoRA fine-tuning support (``--use-lora``) to reduce GPU memory footprint
- Ability to freeze the encoder or run decoder-only updates for quick PoCs
- Custom metrics (CER + exact-match) with per-run JSON logging
- Automatic evaluation on validation and test splits after training

Usage examples (PowerShell)::

    # Fine-tune the kkatiz checkpoint with heavy augmentations + LoRA
    \
    .\.venv\Scripts\python.exe train\train_trocr.py \
        --csv data\tb_match_data_20240705_10581-11080.csv \
        --data-root data\210-20250930T155802Z-1-001 \
        --model-id kkatiz/thai-trocr-thaigov-v2 \
        --output-dir outputs\kkatiz_aug_lora \
        --augment heavy --use-lora --num-train-epochs 12

    # Continue training from your custom checkpoint (directory or .pth state_dict)
    \
    .\.venv\Scripts\python.exe train\train_trocr.py \
        --csv data\tb_match_data_20240705_10581-11080.csv \
        --data-root data\210-20250930T155802Z-1-001 \
        --model-path models\weights\charactor_reader.pth \
        --output-dir outputs\v1_clean --augment none --freeze-encoder

Install requirements (first run)::

    pip install "transformers[torch]" datasets accelerate peft sentencepiece "python-Levenshtein>=0.22"
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from types import MethodType


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    set_seed,
)
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
try:  # transformers>=4.39
    from transformers import DataCollatorForVisionSeq2Seq as HFVisionCollator  # type: ignore
except ImportError:  # pragma: no cover - older transformers fallback
    HFVisionCollator = None  # type: ignore

from train.data_utils import LicensePlateRecord, load_plate_crop, load_records, stratified_split
from train.province_mapping import get_province_token, get_special_tokens, PROVINCE_CODE_TO_THAI_NAME

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - required dependency
    raise ImportError("numpy is required for train_trocr; install with `pip install numpy`." ) from exc

try:
    import Levenshtein  # type: ignore
except ImportError:  # pragma: no cover - optional but faster
    Levenshtein = None  # type: ignore

try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel  # type: ignore
except ImportError:  # pragma: no cover - optional dependency for LoRA
    LoraConfig = TaskType = get_peft_model = PeftModel = None  # type: ignore

logger = logging.getLogger("train_trocr")


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    rotation_degrees: float = 10.0
    blur_prob: float = 0.25
    noise_prob: float = 0.25
    color_jitter_prob: float = 0.5
    occlusion_prob: float = 0.2
    jpeg_prob: float = 0.2
    solarize_prob: float = 0.1


class SimpleAugmentor:
    """Pillow/Numpy based augmentations (keeps dependencies minimal)."""

    def __init__(self, config: AugmentationConfig):
        from PIL import ImageEnhance, ImageFilter, ImageOps

        self.config = config
        self.ImageEnhance = ImageEnhance
        self.ImageFilter = ImageFilter
        self.ImageOps = ImageOps

    def __call__(self, image):
        from PIL import Image

        cfg = self.config
        img = image

        if random.random() < cfg.rotation_degrees / 30.0:  # heuristic probability
            angle = random.uniform(-cfg.rotation_degrees, cfg.rotation_degrees)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=True, fillcolor=tuple(int(x) for x in image.getpixel((0, 0))))

        if random.random() < cfg.blur_prob:
            img = img.filter(self.ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))

        if random.random() < cfg.color_jitter_prob:
            enhancer = self.ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
            enhancer = self.ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.3))

        if random.random() < cfg.solarize_prob:
            img = self.ImageOps.solarize(img, threshold=random.randint(100, 200))

        np_img = np.array(img)

        if random.random() < cfg.noise_prob:
            noise = np.random.normal(0, 10, size=np_img.shape).astype(np.float32)
            np_img = np.clip(np_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() < cfg.occlusion_prob:
            h, w = np_img.shape[:2]
            occ_w = random.randint(max(1, w // 10), max(3, w // 4))
            occ_h = random.randint(max(1, h // 10), max(3, h // 2))
            x0 = random.randint(0, max(0, w - occ_w))
            y0 = random.randint(0, max(0, h - occ_h))
            np_img[y0 : y0 + occ_h, x0 : x0 + occ_w] = random.randint(180, 255)

        if random.random() < cfg.jpeg_prob:
            from io import BytesIO

            buffer = BytesIO()
            Image.fromarray(np_img).save(buffer, format="JPEG", quality=random.randint(30, 70))
            buffer.seek(0)
            img = Image.open(buffer)
            np_img = np.array(img)

        return Image.fromarray(np_img)


def build_augmentor(name: str) -> Optional[Callable]:
    name = name.lower()
    if name in {"none", "off"}:
        return None
    if name in {"light", "baseline"}:
        return SimpleAugmentor(AugmentationConfig(rotation_degrees=5, blur_prob=0.15, noise_prob=0.15, color_jitter_prob=0.3, occlusion_prob=0.1, jpeg_prob=0.1))
    if name in {"medium", "mod"}:
        return SimpleAugmentor(AugmentationConfig(rotation_degrees=10, blur_prob=0.2, noise_prob=0.2, color_jitter_prob=0.4, occlusion_prob=0.15, jpeg_prob=0.15))
    if name in {"heavy", "strong"}:
        return SimpleAugmentor(AugmentationConfig(rotation_degrees=15, blur_prob=0.3, noise_prob=0.3, color_jitter_prob=0.5, occlusion_prob=0.25, jpeg_prob=0.25, solarize_prob=0.15))
    raise ValueError(f"Unknown augmentation preset: {name}")


# ---------------------------------------------------------------------------
# Helper functions for province prediction
# ---------------------------------------------------------------------------


def build_label_with_province(plate_text: str, province_code: Optional[str], province_format: str) -> str:
    """Build training label that includes both plate text and province."""
    if not province_code:
        return plate_text.strip()
    
    province_token = get_province_token(province_code, province_format)
    return f"{plate_text.strip()} <prov> {province_token}"


def split_prediction_and_province(text: str) -> tuple[str, str]:
    """Split predicted text into plate and province parts.
    
    Thai license plates have two parts:
    - Top: License number (e.g., กท2311, 2ฒห5459)
    - Bottom: Province name (e.g., จันทบุรี, กรุงเทพมหานคร)
    """
    import re
    
    # Clean up text first
    text = text.strip()
    
    # Remove common V1 model artifacts
    artifacts = ['President', 'mili', 'ซอง', 'ภรรยาของเขา', 'ขบ', 'cing', 'milimili', 'incing']
    for artifact in artifacts:
        text = text.replace(artifact, ' ')
    
    # Clean up extra spaces
    text = ' '.join(text.split())
    
    # Method 1: Standard format with <prov> token
    if "<prov>" in text:
        parts = text.split("<prov>")
        plate = parts[0].strip()
        province = parts[1].strip() if len(parts) > 1 else ""
        
        # Remove angle brackets from province tokens
        if province.startswith("<") and province.endswith(">"):
            province = province[1:-1]
        
        return plate, province
    
    # Method 2: Look for TH-XX pattern (province codes)
    th_match = re.search(r'(TH-\d+)', text)
    if th_match:
        province_code = th_match.group(1)
        # Remove the province code from text to get plate
        plate = text.replace(province_code, '').strip()
        plate = ' '.join(plate.split())
        return plate, province_code
    
    # Method 3: Look for Thai province names (full names)
    thai_provinces = [
        'กรุงเทพมหานคร', 'สมุทรปราการ', 'นนทบุรี', 'ปทุมธานี', 'พระนครศรีอยุธยา',
        'อ่างทอง', 'ลพบุรี', 'สิงห์บุรี', 'ชัยนาท', 'สระบุรี', 'ชลบุรี', 'ระยอง',
        'จันทบุรี', 'ตราด', 'ฉะเชิงเทรา', 'ปราจีนบุรี', 'นครนายก', 'สระแก้ว',
        'เชียงใหม่', 'ลำพูน', 'ลำปาง', 'อุตรดิตถ์', 'แพร่', 'น่าน', 'พะเยา',
        'เชียงราย', 'แม่ฮ่องสอน', 'นครราชสีมา', 'บุรีรัมย์', 'สุรินทร์', 'ศรีสะเกษ',
        'อุบลราชธานี', 'ยโสธร', 'ชัยภูมิ', 'อำนาจเจริญ', 'หนองบัวลำภู', 'ขอนแก่น',
        'อุดรธานี', 'เลย', 'หนองคาย', 'มหาสารคาม', 'ร้อยเอ็ด', 'กาฬสินธุ์',
        'สกลนคร', 'นครพนม', 'มุกดาหาร', 'ราชบุรี', 'กาญจนบุรี', 'สุพรรณบุรี',
        'นครปฐม', 'สมุทรสาคร', 'สมุทรสงคราม', 'เพชรบุรี', 'ประจวบคีรีขันธ์',
        'นครศรีธรรมราช', 'กระบี่', 'พังงา', 'ภูเก็ต', 'สุราษฎร์ธานี', 'ระนอง',
        'ชุมพร', 'สงขลา', 'สตูล', 'ตรัง', 'พัทลุง', 'ปัตตานี', 'ยะลา', 'นราธิวาส',
        'นครสวรรค์', 'อุทัยธานี', 'กำแพงเพชร', 'ตาก', 'สุโขทัย', 'พิษณุโลก',
        'พิจิตร', 'เพชรบูรณ์'
    ]
    
    # Find the longest matching province name
    found_province = ""
    found_start = -1
    for province in thai_provinces:
        start_pos = text.find(province)
        if start_pos != -1 and len(province) > len(found_province):
            found_province = province
            found_start = start_pos
    
    if found_province:
        # Extract plate part (everything before the province)
        plate = text[:found_start] + text[found_start + len(found_province):]
        plate = plate.strip()
        plate = ' '.join(plate.split())
        return plate, found_province
    
    # Method 4: Extract Thai license plate pattern
    # Thai plates: กท2311, 2ฒห5459, 1กษ1684, etc.
    # Pattern: [Optional digits][Thai consonants][Optional digits]
    plate_matches = re.findall(r'[0-9]*[ก-ฮ]+[0-9]*', text)
    
    if plate_matches:
        # Take the longest/most complete looking plate
        best_plate = max(plate_matches, key=len)
        # Remove it from text to see if there's province info left
        remaining = text.replace(best_plate, '').strip()
        remaining = ' '.join(remaining.split())
        
        # Check if remaining text looks like a province
        if remaining and len(remaining) > 2:
            # If remaining text contains Thai characters, it might be a province
            if re.search(r'[ก-ฮ]', remaining):
                return best_plate, remaining
        
        return best_plate, ""
    
    # Method 5: Split by whitespace and find plate/province parts
    words = text.split()
    if len(words) >= 2:
        # First word usually contains the plate number
        potential_plate = words[0]
        potential_province = ' '.join(words[1:])
        
        # Check if first word looks like a plate (has Thai chars and/or numbers)
        if re.search(r'[ก-ฮ0-9]', potential_plate):
            return potential_plate, potential_province
    
    # Last resort: return the cleaned text as plate
    return text.strip(), ""


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------


class PlateOCRDataset(Dataset):
    def __init__(
        self,
        records: Sequence[LicensePlateRecord],
        processor: TrOCRProcessor,
        *,
        augment_fn: Optional[Callable] = None,
        normalize_text: bool = True,
        predict_province: bool = False,
        province_format: str = "code",
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.augment_fn = augment_fn
        self.normalize_text = normalize_text
        self.predict_province = predict_province
        self.province_format = province_format

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.records)

    def _normalize_text(self, text: str) -> str:
        cleaned = text.strip()
        if self.normalize_text:
            cleaned = cleaned.replace(" ", " ")  # replace non-breaking spaces
        return cleaned

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        image = load_plate_crop(record)
        if self.augment_fn is not None:
            image = self.augment_fn(image)

        # Build label with or without province
        if self.predict_province:
            text = build_label_with_province(
                record.plate_text, 
                record.province_code, 
                self.province_format
            )
        else:
            text = self._normalize_text(record.plate_text)
        
        encoding = self.processor(images=image, text=text, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)
        labels = encoding["labels"].squeeze(0)
        labels = labels.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # CRITICAL: Validate token IDs are within vocabulary range to prevent CUDA index errors
        vocab_size = len(self.processor.tokenizer)
        valid_labels = labels.clone()
        
        # Find problematic tokens
        mask_invalid = (valid_labels >= vocab_size) | (valid_labels < -100)
        if torch.any(mask_invalid):
            logger.warning(f"Sample {idx}: Found {torch.sum(mask_invalid).item()} invalid token IDs. Max token: {torch.max(valid_labels).item()}, vocab_size: {vocab_size}")
            # Replace invalid tokens with pad token ID or clamp to valid range
            valid_labels = torch.where(
                mask_invalid & (valid_labels >= 0),  # Only clamp positive invalid tokens
                torch.tensor(self.processor.tokenizer.pad_token_id, dtype=valid_labels.dtype),
                valid_labels
            )
            # Ensure all positive tokens are within vocabulary
            valid_labels = torch.clamp(valid_labels, -100, vocab_size - 1)

        batch: Dict[str, torch.Tensor] = {
            "pixel_values": pixel_values,
            "labels": valid_labels,
        }
        if "decoder_attention_mask" in encoding:
            batch["decoder_attention_mask"] = encoding["decoder_attention_mask"].squeeze(0)
        return batch


# ---------------------------------------------------------------------------
# Data collator fallback
# ---------------------------------------------------------------------------


@dataclass
class VisionSeq2SeqCollator:
    processor: TrOCRProcessor

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        label_pad_token_id = self.processor.tokenizer.pad_token_id

        pixel_values = torch.stack([f["pixel_values"] for f in features])

        label_tensors: List[torch.Tensor] = []
        for feature in features:
            label = feature["labels"].clone()
            if label_pad_token_id is not None:
                label[label == -100] = label_pad_token_id
            label_tensors.append(label)

        labels = torch.nn.utils.rnn.pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=label_pad_token_id if label_pad_token_id is not None else 0,
        )
        if label_pad_token_id is not None:
            labels[labels == label_pad_token_id] = -100

        batch: Dict[str, torch.Tensor] = {"pixel_values": pixel_values, "labels": labels}

        if "decoder_attention_mask" in features[0]:
            dam = torch.nn.utils.rnn.pad_sequence(
                [f["decoder_attention_mask"] for f in features],
                batch_first=True,
                padding_value=0,
            )
            batch["decoder_attention_mask"] = dam

        return batch


class VisionSeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer variant that strips text-centric keys before forwarding.

    This avoids accidentally passing `input_ids` to the ViT encoder when using
    LoRA or other adapters that expect text inputs.
    """

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prepared = super()._prepare_inputs(inputs)
        prepared.pop("input_ids", None)
        prepared.pop("attention_mask", None)
        return prepared

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        inputs = self._prepare_inputs(inputs)
        pixel_values = inputs.pop("pixel_values")
        labels = inputs.pop("labels")
        decoder_attention_mask = inputs.pop("decoder_attention_mask", None)

        if inputs:
            logger.info("VisionSeq2SeqTrainer dropping extra inputs during loss: %s", list(inputs.keys()))

        # CRITICAL: Additional validation to prevent CUDA index errors
        if labels is not None:
            vocab_size = model.config.decoder.vocab_size if hasattr(model.config, 'decoder') else model.config.vocab_size
            if vocab_size is None:
                vocab_size = model.decoder.get_input_embeddings().num_embeddings
            
            # Check for out-of-bounds tokens
            valid_mask = (labels >= -100) & (labels < vocab_size)
            invalid_mask = ~valid_mask & (labels >= 0)  # Only check positive tokens
            
            if torch.any(invalid_mask):
                num_invalid = torch.sum(invalid_mask).item()
                max_token = torch.max(labels[labels >= 0]).item() if torch.any(labels >= 0) else -1
                logger.error(f"CRITICAL: Found {num_invalid} invalid token IDs during training. Max token: {max_token}, vocab_size: {vocab_size}")
                
                # Emergency fix: clamp all positive tokens to valid range
                labels = torch.where(
                    invalid_mask,
                    torch.tensor(vocab_size - 1, dtype=labels.dtype, device=labels.device),
                    labels
                )
                logger.warning("Emergency token clamping applied to prevent CUDA crash")

        model_kwargs = {
            "pixel_values": pixel_values,
            "labels": labels,
        }
        if decoder_attention_mask is not None:
            model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        logger.info("VisionSeq2SeqTrainer model kwargs (loss): %s", list(model_kwargs.keys()))
        outputs = model(**model_kwargs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(  # type: ignore[override]
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys=None,
    ):
        inputs = self._prepare_inputs(inputs)
        pixel_values = inputs.pop("pixel_values")
        labels = inputs.get("labels")
        decoder_attention_mask = inputs.pop("decoder_attention_mask", None)

        with torch.no_grad():
            if inputs:
                logger.info("VisionSeq2SeqTrainer dropping extra inputs during pred: %s", list(inputs.keys()))

            # Check tensor dimensions to prevent CUDA errors
            if pixel_values is not None:
                logger.debug(f"pixel_values shape: {pixel_values.shape}")
            if labels is not None:
                logger.debug(f"labels shape: {labels.shape}")
                # Ensure labels are within vocab range
                vocab_size = model.config.decoder.vocab_size if hasattr(model.config, 'decoder') else model.config.vocab_size
                if torch.any(labels >= vocab_size):
                    logger.warning(f"Labels contain out-of-range tokens. Max label: {labels.max()}, vocab_size: {vocab_size}")
                    # Clamp labels to valid range
                    labels = torch.clamp(labels, 0, vocab_size - 1)

            model_kwargs = {
                "pixel_values": pixel_values,
                "labels": labels,
            }
            if decoder_attention_mask is not None:
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

            logger.info("VisionSeq2SeqTrainer model kwargs (pred): %s", list(model_kwargs.keys()))
            
            # Clear GPU memory before prediction for better performance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            try:
                # Use torch.inference_mode for better performance during evaluation
                with torch.inference_mode():
                    outputs = model(**model_kwargs)
            except RuntimeError as e:
                if "CUDA error" in str(e):
                    logger.error(f"CUDA error during prediction: {e}")
                    # Try to recover by reducing memory usage
                    torch.cuda.empty_cache()
                    # Skip this batch
                    return (None, None, None)
                else:
                    raise

        if prediction_loss_only:
            return (outputs.loss, None, None)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        loss = outputs.loss if hasattr(outputs, "loss") else None
        return (loss, logits, labels)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def character_error_rate(prediction: str, reference: str) -> float:
    prediction = prediction or ""
    reference = reference or ""
    if not reference and not prediction:
        return 0.0
    if not reference:
        return float(len(prediction))
    if Levenshtein:
        dist = Levenshtein.distance(prediction, reference)  # type: ignore[attr-defined]
    else:
        dist = _levenshtein_distance(prediction, reference)
    return dist / max(1, len(reference))


def _levenshtein_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current_row = [i]
        for j, cb in enumerate(b, 1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Thai TrOCR models on plate crops")
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV (tb_match_...) ")
    parser.add_argument("--data-root", type=str, action="append", required=True, help="Root directory containing the /210/... hierarchy. Use multiple times for extra roots.")
    parser.add_argument("--model-id", type=str, default="kkatiz/thai-trocr-thaigov-v2", help="HuggingFace model identifier to fine-tune.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional local directory or .pth state_dict to initialize from instead of --model-id.")
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save checkpoints and logs.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size-reduction", type=int, default=1, help="Reduce eval batch size by this factor for speed (1=no reduction, 2=half size, etc.)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--num-train-epochs", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--augment", type=str, default="light", choices=["none", "light", "medium", "heavy", "baseline", "mod", "strong"])
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze encoder weights (only train decoder).")
    parser.add_argument("--decoder-only", action="store_true", help="Alias for --freeze-encoder (for clarity).")
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning (requires peft).")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="auto",
        help=(
            "Comma-separated module names for LoRA injection, or 'auto' to detect based on the architecture "
            "(Electra/CamemBERT → query,key,value,dense; LLaMA/GPTQ → q_proj,k_proj,v_proj,o_proj)."
        ),
    )

    # Ensure save_steps is a round multiple of eval_steps when load_best_model_at_end is used.
    try:
        load_best = False if args.eval_only_at_end else bool(len(datasets["val"]))
        if load_best and training_args.eval_steps and training_args.save_steps:
            if training_args.save_strategy == "steps":
                if training_args.save_steps % training_args.eval_steps != 0:
                    # Adjust save_steps to be the nearest multiple of eval_steps >= current save_steps
                    multiples = (training_args.save_steps + training_args.eval_steps - 1) // training_args.eval_steps
                    new_save = multiples * training_args.eval_steps
                    logger.warning(
                        "Adjusting save_steps from %d to %d to be a multiple of eval_steps=%d so load_best_model_at_end works.",
                        training_args.save_steps,
                        new_save,
                        training_args.eval_steps,
                    )
                    training_args.save_steps = new_save
    except Exception:
        # Defensive: if something unexpected happens, proceed with defaults and let transformers validate
        pass
    parser.add_argument(
        "--lora-scope",
        type=str,
        choices=["decoder", "encoder", "all"],
        default="decoder",
        help="Restrict trainable LoRA params to decoder, encoder, or both (applies requires_grad filters post-injection).",
    )
    parser.add_argument("--fp16", action="store_true", help="Use float16 mixed precision.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision (Ampere+ GPUs).")
    parser.add_argument("--eval-steps", type=int, default=2000, help="Validation frequency (higher = less frequent, faster training)")
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--eval-every-epoch", action="store_true", help="Run evaluation at the end of every epoch instead of steps")
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--eval-only-at-end", action="store_true", help="Skip validation during training, only evaluate at the end for speed")
    parser.add_argument("--report-to", type=str, nargs="*", default=None, help="e.g. wandb tensorboard")
    parser.add_argument("--generation-max-length", type=int, default=32)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--manifest-dir", type=str, default=None, help="Optional directory to dump train/val/test manifests (jsonl).")
    parser.add_argument("--only-eval", action="store_true", help="Skip training and only run evaluation (requires --resume-from-checkpoint or existing output_dir).")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers (0 is safest on Windows; increase on Linux for speed).",
    )
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda).")
    parser.add_argument("--predict-province", action="store_true", help="Include province prediction in training targets.")
    parser.add_argument("--province-format", type=str, default="code", choices=["code", "name"], help="Use province code (TH-XX) or Thai name.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model preparation
# ---------------------------------------------------------------------------


def load_model_and_processor(args: argparse.Namespace) -> tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
    model_id = args.model_id

    processor_path: Optional[Path] = None
    if getattr(args, "model_path", None):
        mp = Path(args.model_path)
        if mp.is_dir():
            candidate = mp / "processor"
            if candidate.exists():
                processor_path = candidate

    processor_source = processor_path if processor_path is not None else model_id
    processor = TrOCRProcessor.from_pretrained(processor_source)

    # Add province tokens if predict_province is enabled and not already present
    if getattr(args, "predict_province", False):
        tokenizer = processor.tokenizer
        province_tokens = get_special_tokens(args.province_format)
        
        # Check if tokens are already in the tokenizer vocabulary
        existing_tokens = set(tokenizer.get_vocab().keys())
        missing_tokens = [token for token in province_tokens if token not in existing_tokens]
        
        if missing_tokens:
            added = tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
            if added > 0:
                logger.info("Added %d province tokens to tokenizer", added)
        else:
            logger.info("All %d province tokens already present in tokenizer", len(province_tokens))

    if getattr(args, "model_path", None):
        source_path = Path(args.model_path)
        if source_path.is_dir():
            adapter_candidates: list[Path] = [
                source_path / "adapter_model.safetensors",
                source_path / "adapter_model.bin",
                source_path / "adapter_model.pt",
            ]
            adapter_path = next((candidate for candidate in adapter_candidates if candidate.exists()), None)
            adapter_exists = adapter_path is not None
            use_lora = bool(getattr(args, "use_lora", False) or adapter_exists)
            if adapter_exists and not getattr(args, "use_lora", False):
                logger.info(
                    "Detected LoRA adapter weights (%s) in %s; enabling use_lora for loading.",
                    adapter_path.name,
                    source_path,
                )
            if use_lora:
                if not adapter_exists:
                    available = ", ".join(p.name for p in source_path.glob("adapter_model.*")) or "none"
                    raise RuntimeError(
                        "use_lora=True but no adapter_model.[safetensors|bin|pt] found in %s (found: %s)"
                        % (source_path, available)
                    )
                if PeftModel is None:
                    raise RuntimeError("peft is required to load LoRA adapters. Install with `pip install peft`.")
                logger.info("Loading base %s with LoRA adapters from %s", model_id, source_path)
                base_model = VisionEncoderDecoderModel.from_pretrained(model_id)
                
                # Resize embeddings if needed BEFORE loading LoRA
                if getattr(args, "predict_province", False):
                    new_vocab_size = len(processor.tokenizer)
                    # Ensure vocab size accounts for highest token ID + 1
                    vocab = processor.tokenizer.get_vocab()
                    max_token_id = max(vocab.values()) if vocab else 0
                    required_vocab_size = max(new_vocab_size, max_token_id + 1)
                    
                    if base_model.decoder.get_input_embeddings().num_embeddings != required_vocab_size:
                        logger.info("Resizing base model decoder embeddings from %d to %d tokens (tokenizer len: %d, max_id: %d)", 
                                   base_model.decoder.get_input_embeddings().num_embeddings, required_vocab_size, new_vocab_size, max_token_id)
                        base_model.decoder.resize_token_embeddings(required_vocab_size)
                        base_model.config.decoder.vocab_size = required_vocab_size
                
                model = PeftModel.from_pretrained(base_model, source_path)
            else:
                logger.info("Loading model from directory %s", source_path)
                model = VisionEncoderDecoderModel.from_pretrained(source_path)
                
                # Check if we need to resize embeddings for saved model
                if getattr(args, "predict_province", False):
                    new_vocab_size = len(processor.tokenizer)
                    # Ensure vocab size accounts for highest token ID + 1
                    vocab = processor.tokenizer.get_vocab()
                    max_token_id = max(vocab.values()) if vocab else 0
                    required_vocab_size = max(new_vocab_size, max_token_id + 1)
                    
                    if model.decoder.get_input_embeddings().num_embeddings != required_vocab_size:
                        logger.info("Resizing loaded model decoder embeddings from %d to %d tokens (tokenizer len: %d, max_id: %d)", 
                                   model.decoder.get_input_embeddings().num_embeddings, required_vocab_size, new_vocab_size, max_token_id)
                        model.decoder.resize_token_embeddings(required_vocab_size)
                        model.config.decoder.vocab_size = required_vocab_size
        else:
            logger.info("Loading %s and applying state_dict from %s", model_id, source_path)
            model = VisionEncoderDecoderModel.from_pretrained(model_id)
            
            # Load state dict first
            state = torch.load(source_path, map_location="cpu")
            if isinstance(state, dict) and any(key.startswith("module.") for key in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            
            # For province prediction with mismatched vocab sizes, we need special handling
            if getattr(args, "predict_province", False):
                new_vocab_size = len(processor.tokenizer)
                vocab = processor.tokenizer.get_vocab()
                max_token_id = max(vocab.values()) if vocab else 0
                required_vocab_size = max(new_vocab_size, max_token_id + 1)
                
                # Get current model embedding size and state dict embedding size
                current_model_size = model.decoder.get_input_embeddings().num_embeddings
                
                # Find embedding tensors in state dict to get their size
                state_vocab_size = None
                for key in state.keys():
                    if "embeddings.word_embeddings.weight" in key or "lm_head.weight" in key:
                        state_vocab_size = state[key].shape[0]
                        break
                
                if state_vocab_size and state_vocab_size != required_vocab_size:
                    logger.info("Vocab size mismatch: state_dict=%d, current_model=%d, required=%d", 
                               state_vocab_size, current_model_size, required_vocab_size)
                    
                    # First load what we can from the state dict (embedding layers will be skipped due to size mismatch)
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    
                    # Then resize to the required size
                    logger.info("Resizing model decoder embeddings from %d to %d tokens after partial state_dict load", 
                               model.decoder.get_input_embeddings().num_embeddings, required_vocab_size)
                    model.decoder.resize_token_embeddings(required_vocab_size)
                    model.config.decoder.vocab_size = required_vocab_size
                    
                    # Copy over the original embeddings that we can
                    if state_vocab_size < required_vocab_size:
                        logger.info("Copying %d original embeddings to resized model", state_vocab_size)
                        for key in state.keys():
                            if "embeddings.word_embeddings.weight" in key:
                                # Copy the original embeddings
                                current_embeddings = model.decoder.get_input_embeddings().weight.data
                                current_embeddings[:state_vocab_size] = state[key]
                            elif "lm_head.weight" in key or "generator_lm_head.weight" in key:
                                # Find the corresponding layer in the model
                                target_param = None
                                for name, param in model.named_parameters():
                                    if "lm_head.weight" in name or "generator_lm_head.weight" in name:
                                        target_param = param
                                        break
                                if target_param is not None:
                                    target_param.data[:state_vocab_size] = state[key]
                            elif "lm_head.bias" in key or "generator_lm_head.bias" in key:
                                # Copy bias terms
                                for name, param in model.named_parameters():
                                    if "lm_head.bias" in name or "generator_lm_head.bias" in name:
                                        param.data[:state_vocab_size] = state[key]
                                        break
                else:
                    # Sizes match or no embedding layers found, normal loading
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    if model.decoder.get_input_embeddings().num_embeddings != required_vocab_size:
                        logger.info("Resizing model decoder embeddings from %d to %d tokens after state_dict load", 
                                   model.decoder.get_input_embeddings().num_embeddings, required_vocab_size)
                        model.decoder.resize_token_embeddings(required_vocab_size)
                        model.config.decoder.vocab_size = required_vocab_size
            else:
                # Normal loading without province tokens
                missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning("Missing keys when loading state_dict: %s", missing[:8])
            if unexpected:
                logger.warning("Unexpected keys when loading state_dict: %s", unexpected[:8])
    else:
        model = VisionEncoderDecoderModel.from_pretrained(model_id)
        
        # Resize embeddings for fresh model if needed
        if getattr(args, "predict_province", False):
            new_vocab_size = len(processor.tokenizer)
            # Ensure vocab size accounts for highest token ID + 1
            vocab = processor.tokenizer.get_vocab()
            max_token_id = max(vocab.values()) if vocab else 0
            required_vocab_size = max(new_vocab_size, max_token_id + 1)
            
            if model.decoder.get_input_embeddings().num_embeddings != required_vocab_size:
                logger.info("Resizing fresh model decoder embeddings from %d to %d tokens (tokenizer len: %d, max_id: %d)", 
                           model.decoder.get_input_embeddings().num_embeddings, required_vocab_size, new_vocab_size, max_token_id)
                model.decoder.resize_token_embeddings(required_vocab_size)
                model.config.decoder.vocab_size = required_vocab_size

    tokenizer = processor.tokenizer
    model.config.decoder_start_token_id = tokenizer.cls_token_id or tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Validate and fix vocab size mismatches
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = model.decoder.get_input_embeddings().num_embeddings
    
    if tokenizer_vocab_size != model_vocab_size:
        logger.warning(f"Vocab size mismatch: tokenizer={tokenizer_vocab_size}, model={model_vocab_size}")
        # Use the larger of the two to be safe
        safe_vocab_size = max(tokenizer_vocab_size, model_vocab_size)
        if model_vocab_size < safe_vocab_size:
            logger.info(f"Resizing model vocab from {model_vocab_size} to {safe_vocab_size}")
            model.decoder.resize_token_embeddings(safe_vocab_size)
        model.config.vocab_size = safe_vocab_size
        model.config.decoder.vocab_size = safe_vocab_size
    else:
        model.config.vocab_size = tokenizer_vocab_size
        
    model.config.max_length = args.generation_max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 2

    if args.freeze_encoder or args.decoder_only:
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen; training decoder-only.")

    already_peft = PeftModel is not None and isinstance(model, PeftModel)

    if args.use_lora and not already_peft:
        if get_peft_model is None:
            raise RuntimeError("peft is not installed. Install with `pip install peft`." )
        target_modules = _resolve_lora_targets(args, model)
        lora_task_type = TaskType.SEQ_2_SEQ_LM if TaskType is not None else None
        if TaskType is not None and hasattr(TaskType, "MULTI_MODAL_ENCODER_DECODER"):
            lora_task_type = TaskType.MULTI_MODAL_ENCODER_DECODER  # type: ignore[attr-defined]

        lora_config = LoraConfig(
            task_type=lora_task_type,  # type: ignore[arg-type]
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Limit trainable params to the requested scope
        if args.lora_scope != "all":
            for name, param in model.named_parameters():
                is_lora = ("lora_A" in name) or ("lora_B" in name) or ("lora_embedding" in name)
                if not is_lora:
                    continue
                if args.lora_scope == "decoder" and not name.startswith("base_model.model.decoder") and not name.startswith("model.decoder"):
                    param.requires_grad = False
                if args.lora_scope == "encoder" and not name.startswith("base_model.model.encoder") and not name.startswith("model.encoder"):
                    param.requires_grad = False
    elif args.use_lora and already_peft:
        logger.info("LoRA adapters already active on model; skipping reinjection.")

    # Resize token embeddings if province tokens were added
    if getattr(args, "predict_province", False):
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        logger.info("Resized decoder embeddings to %d tokens", len(processor.tokenizer))

    _patch_encoder_forward(model)
    return model, processor


def _resolve_lora_targets(args: argparse.Namespace, model: VisionEncoderDecoderModel) -> List[str]:
    """Choose sensible LoRA target modules based on the decoder/encoder architecture.

    - For Electra/Roberta/Camembert style decoders: query,key,value,dense
    - For LLaMA/GPT-NeoX/BART style: q_proj,k_proj,v_proj,o_proj
    - If user provides an explicit comma-separated list, use that.
    """
    user = getattr(args, "lora_target_modules", "auto")
    if user and user.lower() != "auto":
        return [m.strip() for m in user.split(",") if m.strip()]

    dec = getattr(model, "decoder", None)
    dec_cls = dec.__class__.__name__.lower() if dec is not None else ""
    enc = getattr(model, "encoder", None)
    enc_cls = enc.__class__.__name__.lower() if enc is not None else ""

    # Electra/Roberta/Camembert families expose attention blocks with query/key/value/dense
    if any(k in dec_cls for k in ["electra", "roberta", "camembert"]):
        return ["query", "key", "value", "dense"]

    # Fallback to common q_proj/v_proj/o_proj used in many decoder architectures
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def _patch_encoder_forward(model: VisionEncoderDecoderModel) -> None:
    """Ensure the vision encoder ignores unexpected `input_ids` kwargs."""

    def _wrap(encoder: Any) -> None:
        if encoder is None:
            return
        original_forward = getattr(encoder, "forward", None)
        original_forward_fn = getattr(original_forward, "__func__", None)
        if original_forward_fn is None:
            original_forward_fn = getattr(encoder.__class__, "forward", None)
        if original_forward_fn is None:
            return
        if getattr(original_forward_fn, "_ignores_input_ids", False):
            return

        allowed = {
            "pixel_values",
            "head_mask",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "interpolate_pos_encoding",
        }

        @functools.wraps(original_forward_fn)
        def wrapped_forward(self, *args, **kwargs):
            if kwargs:
                kwargs = {k: v for k, v in kwargs.items() if k in allowed}
            if args and kwargs and "pixel_values" in kwargs:
                kwargs = dict(kwargs)
                kwargs.pop("pixel_values")
            return original_forward_fn(self, *args, **kwargs)

        setattr(wrapped_forward, "_ignores_input_ids", True)
        encoder.forward = MethodType(wrapped_forward, encoder)

    # Base encoder (no LoRA)
    _wrap(getattr(model, "encoder", None))

    # LoRA models nest the actual encoder under base_model.model.encoder
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        inner_model = getattr(base_model, "model", None)
        if inner_model is not None:
            _wrap(getattr(inner_model, "encoder", None))


# ---------------------------------------------------------------------------
# Trainer assembly
# ---------------------------------------------------------------------------


def build_datasets(
    splits: Dict[str, Sequence[LicensePlateRecord]],
    processor: TrOCRProcessor,
    args: argparse.Namespace,
) -> Dict[str, PlateOCRDataset]:
    augment_fn = build_augmentor(args.augment)

    def maybe_trim(records: Sequence[LicensePlateRecord], limit: Optional[int]) -> List[LicensePlateRecord]:
        if limit is None or limit >= len(records):
            return list(records)
        return list(records)[:limit]

    datasets: Dict[str, PlateOCRDataset] = {}
    datasets["train"] = PlateOCRDataset(
        maybe_trim(splits.get("train", []), args.max_train_samples),
        processor,
        augment_fn=augment_fn,
        normalize_text=True,
        predict_province=getattr(args, "predict_province", False),
        province_format=getattr(args, "province_format", "code"),
    )
    datasets["val"] = PlateOCRDataset(
        maybe_trim(splits.get("val", []), args.max_eval_samples),
        processor,
        augment_fn=None,
        normalize_text=True,
        predict_province=getattr(args, "predict_province", False),
        province_format=getattr(args, "province_format", "code"),
    )
    datasets["test"] = PlateOCRDataset(
        maybe_trim(splits.get("test", []), args.max_eval_samples),
        processor,
        augment_fn=None,
        normalize_text=True,
        predict_province=getattr(args, "predict_province", False),
        province_format=getattr(args, "province_format", "code"),
    )
    return datasets


def compute_metrics_builder(processor: TrOCRProcessor, predict_province: bool = False) -> Callable:
    pad_token_id = processor.tokenizer.pad_token_id

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        label_ids = np.asarray(labels)

        if predictions.ndim == 3:  # (batch, beams, seq)
            predictions = predictions[:, 0, :]
        if label_ids.ndim == 3:
            label_ids = label_ids[:, 0, :]

        predictions = predictions.astype(np.int64, copy=False)
        label_ids = label_ids.astype(np.int64, copy=False)

        predictions[predictions == -100] = pad_token_id
        label_ids[label_ids == -100] = pad_token_id
        predictions[predictions < 0] = pad_token_id
        label_ids[label_ids < 0] = pad_token_id

        pred_texts = processor.batch_decode(predictions, skip_special_tokens=True)
        label_texts = processor.batch_decode(label_ids, skip_special_tokens=True)

        if predict_province:
            # Separate metrics for plate text and province
            plate_cer_scores = []
            province_cer_scores = []
            plate_exact = 0
            province_exact = 0
            combined_exact = 0
            
            for pred, label in zip(pred_texts, label_texts):
                pred_plate, pred_province = split_prediction_and_province(pred)
                label_plate, label_province = split_prediction_and_province(label)
                
                # Plate metrics
                plate_cer = character_error_rate(pred_plate, label_plate)
                plate_cer_scores.append(plate_cer)
                if pred_plate.strip() == label_plate.strip():
                    plate_exact += 1
                
                # Province metrics  
                province_cer = character_error_rate(pred_province, label_province)
                province_cer_scores.append(province_cer)
                if pred_province.strip() == label_province.strip():
                    province_exact += 1
                
                # Combined exact match
                if pred.strip() == label.strip():
                    combined_exact += 1
            
            total_samples = max(1, len(pred_texts))
            return {
                "cer": float(sum(plate_cer_scores) / max(1, len(plate_cer_scores))),
                "exact_match": float(combined_exact / total_samples),
                "plate_cer": float(sum(plate_cer_scores) / max(1, len(plate_cer_scores))),
                "plate_exact_match": float(plate_exact / total_samples),
                "province_cer": float(sum(province_cer_scores) / max(1, len(province_cer_scores))),
                "province_exact_match": float(province_exact / total_samples),
            }
        else:
            # Original metrics for plate text only
            cer_scores = [character_error_rate(p, l) for p, l in zip(pred_texts, label_texts)]
            exact = sum(1 for p, l in zip(pred_texts, label_texts) if p.strip() == l.strip()) / max(1, len(pred_texts))
            return {
                "cer": float(sum(cer_scores) / max(1, len(cer_scores))),
                "exact_match": float(exact),
            }

    return compute_metrics


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    try:
        import accelerate  # noqa: F401
    except ImportError as exc:  # pragma: no cover - informative message before Trainer raises
        raise ImportError(
            "accelerate must be installed for seq2seq training; install with `pip install accelerate`."
        ) from exc

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logger.info("Loading records from %s", args.csv)

    records = load_records(
        args.csv,
        data_roots=args.data_root,
        require_validate=True,
        drop_missing_images=True,
    )
    if not records:
        raise RuntimeError("No records loaded. Check CSV path and --data-root values.")

    if not math.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0, rel_tol=1e-3):
        raise ValueError("train/val/test ratios must sum to 1.0")

    logger.info("Creating stratified splits (train=%.2f, val=%.2f, test=%.2f)", args.train_ratio, args.val_ratio, args.test_ratio)
    splits = stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if args.manifest_dir:
        from train.data_utils import export_manifest

        export_manifest(splits, args.manifest_dir, include_metadata=True)

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / Path(args.model_path or args.model_id).name
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    set_seed(args.seed)

    model, processor = load_model_and_processor(args)
    datasets = build_datasets(splits, processor, args)

    # Choose evaluation/save strategy: epoch-based if requested, otherwise steps-based (if val set)
    if args.eval_only_at_end:
        eval_strategy = "no"
    elif args.eval_every_epoch:
        eval_strategy = "epoch"
    else:
        eval_strategy = "steps" if len(datasets["val"]) else "no"

    save_strategy = "epoch" if args.eval_every_epoch and len(datasets["val"]) else ("steps" if len(datasets["val"]) else "epoch")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=max(1, args.per_device_eval_batch_size // args.eval_batch_size_reduction),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_strategy=save_strategy,
        save_steps=args.save_steps,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        generation_num_beams=1,  # Reduce complexity to avoid CUDA errors
        push_to_hub=args.push_to_hub,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=args.report_to,
        load_best_model_at_end=False if args.eval_only_at_end else (False if args.eval_every_epoch else bool(len(datasets["val"]))),
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=False,  # Reduce memory pressure
        skip_memory_metrics=True,     # Skip memory tracking
    )

    # Always use the local collator to avoid passing text input_ids to the ViT encoder.
    data_collator = VisionSeq2SeqCollator(processor=processor)
    compute_metrics = compute_metrics_builder(processor, getattr(args, "predict_province", False))

    trainer = VisionSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"] if not args.only_eval else None,
        eval_dataset=datasets["val"] if len(datasets["val"]) else None,
        tokenizer=processor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if not args.only_eval:
        logger.info("Starting training: %d train samples, %d val samples", len(datasets["train"]), len(datasets["val"]))
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
        # Save model and config properly
        trainer.save_model()
        # Explicitly save the model config to ensure encoder/decoder configs are preserved
        model.config.save_pretrained(output_dir)
        processor.save_pretrained(output_dir / "processor")

    if len(datasets["val"]):
        logger.info("Evaluating on validation set")
        val_metrics = trainer.evaluate(eval_dataset=datasets["val"], metric_key_prefix="val")
        _dump_metrics(val_metrics, output_dir / "val_metrics.json")

    if len(datasets["test"]):
        logger.info("Evaluating on test set")
        test_metrics = trainer.evaluate(eval_dataset=datasets["test"], metric_key_prefix="test")
        _dump_metrics(test_metrics, output_dir / "test_metrics.json")

    logger.info("All done. Checkpoints saved to %s", output_dir)


def _dump_metrics(metrics: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)
    logger.info("Saved metrics to %s", path)


if __name__ == "__main__":  # pragma: no cover
    main()
