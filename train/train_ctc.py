"""Command-line training script for CTC-based Thai License Plate OCR models.

CTC (Connectionist Temporal Classification) is more data-efficient than Transformer-based
models like TrOCR, making it ideal for limited datasets. This script trains a CNN+RNN+CTC
architecture that can recognize both plate text and province names.

Key features
------------
- Lightweight CNN backbone (ResNet or EfficientNet) for feature extraction
- Bidirectional LSTM for sequence modeling
- CTC loss for alignment-free training
- Support for both plate-only and plate+province prediction
- Data augmentation compatible with small datasets
- Much faster training and inference than TrOCR
- Lower GPU memory requirements

Architecture
------------
Input Image (H x W x 3)
    ↓
CNN Backbone (ResNet18/34 or EfficientNet-B0)
    ↓
Feature Map (C x H' x W')
    ↓
Reshape to Sequence (W' x C*H')
    ↓
Bidirectional LSTM (2-3 layers)
    ↓
Fully Connected Layer
    ↓
CTC Loss

Usage examples (PowerShell)::

    # Train CTC model on plate data
    python train/train_ctc.py \
        --csv data/tb_match_data_20240705_10581-11080.csv \
        --data-root data/210-20250930T155802Z-1-001 \
        --output-dir outputs/ctc_base \
        --backbone resnet18 \
        --num-train-epochs 50

    # Train with province prediction
    python train/train_ctc.py \
        --csv data/combined_plates.csv \
        --data-root data/210-20250930T155802Z-1-001 \
        --output-dir outputs/ctc_with_province \
        --backbone efficientnet_b0 \
        --predict-province \
        --num-train-epochs 80

    # Use synthetic data
    python train/train_ctc.py \
        --csv synthetic_plates/synthetic_plates.csv \
        --data-root synthetic_plates \
        --output-dir outputs/ctc_synthetic \
        --backbone resnet34 \
        --augment heavy

Install requirements::

    pip install torch torchvision timm pillow opencv-python numpy python-Levenshtein
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

try:
    import torchvision.models as models
except ImportError:
    models = None

try:
    import timm  # For EfficientNet and other modern backbones
except ImportError:
    timm = None

try:
    import numpy as np
except ImportError:
    raise ImportError("numpy is required; install with `pip install numpy`")

try:
    import Levenshtein
except ImportError:
    Levenshtein = None

from train.data_utils import LicensePlateRecord, load_plate_crop, load_records, stratified_split
from train.province_mapping import get_province_token, get_special_tokens, PROVINCE_CODE_TO_THAI_NAME

logger = logging.getLogger("train_ctc")


# ---------------------------------------------------------------------------
# Character set and encoding
# ---------------------------------------------------------------------------

# Thai consonants used in license plates
THAI_CHARS = [
    "ก", "ข", "ค", "ฆ", "ง", "จ", "ฉ", "ช", "ซ", "ฌ", "ญ",
    "ฎ", "ฏ", "ฐ", "ฑ", "ฒ", "ณ", "ด", "ต", "ถ", "ท",
    "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ", "ฟ", "ภ", "ม",
    "ย", "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"
]

# Digits
DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Special tokens
SPACE = " "
CTC_BLANK = "<blank>"
PROVINCE_SEP = "<prov>"


class CharacterEncoder:
    """Encode/decode text to indices for CTC training."""
    
    def __init__(self, include_province: bool = False, province_format: str = "code"):
        self.include_province = include_province
        self.province_format = province_format
        
        # Build character set
        chars = [CTC_BLANK, SPACE] + THAI_CHARS + DIGITS
        
        # Add special characters used in tokens (for <prov>, <TH-XX>, etc.)
        special_chars = ["<", ">", "-", "p", "r", "o", "v", "T", "H"]
        for c in special_chars:
            if c not in chars:
                chars.append(c)
        
        # Add province tokens if needed
        if include_province:
            # PROVINCE_SEP is already added via special_chars
            province_tokens = get_special_tokens(province_format)
            # For province names, add all unique Thai characters
            if province_format == "name":
                province_chars = set()
                for province_name in PROVINCE_CODE_TO_THAI_NAME.values():
                    province_chars.update(province_name)
                # Add only new characters not already in THAI_CHARS
                new_chars = sorted(province_chars - set(THAI_CHARS) - set(chars))
                chars.extend(new_chars)
        
        self.chars = chars
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.blank_idx = 0
        
    def __len__(self) -> int:
        return len(self.chars)
    
    def encode(self, text: str) -> List[int]:
        """Convert text to list of character indices."""
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Unknown character - log warning but continue
                logger.warning(f"Unknown character '{char}' in text: {text}")
        return indices
    
    def decode(self, indices: List[int], remove_duplicates: bool = True) -> str:
        """Convert list of indices to text, applying CTC decoding rules."""
        chars = []
        prev_idx = None
        
        for idx in indices:
            # Skip blanks
            if idx == self.blank_idx:
                prev_idx = None
                continue
            
            # Skip duplicates if CTC decoding
            if remove_duplicates and idx == prev_idx:
                continue
            
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
            
            prev_idx = idx
        
        return "".join(chars)
    
    def decode_batch(self, batch_indices: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> List[str]:
        """Decode a batch of predictions."""
        results = []
        batch_size = batch_indices.size(0)
        
        for i in range(batch_size):
            if lengths is not None:
                length = lengths[i].item()
                indices = batch_indices[i, :length].tolist()
            else:
                indices = batch_indices[i].tolist()
            
            text = self.decode(indices)
            results.append(text)
        
        return results


# ---------------------------------------------------------------------------
# Augmentation (reuse from train_trocr.py)
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
    """Pillow/Numpy based augmentations."""

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

        if random.random() < cfg.rotation_degrees / 30.0:
            angle = random.uniform(-cfg.rotation_degrees, cfg.rotation_degrees)
            img = img.rotate(angle, resample=Image.BILINEAR, expand=True, 
                           fillcolor=tuple(int(x) for x in image.getpixel((0, 0))))

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
        return SimpleAugmentor(AugmentationConfig(
            rotation_degrees=5, blur_prob=0.15, noise_prob=0.15, 
            color_jitter_prob=0.3, occlusion_prob=0.1, jpeg_prob=0.1))
    if name in {"medium", "mod"}:
        return SimpleAugmentor(AugmentationConfig(
            rotation_degrees=10, blur_prob=0.2, noise_prob=0.2, 
            color_jitter_prob=0.4, occlusion_prob=0.15, jpeg_prob=0.15))
    if name in {"heavy", "strong"}:
        return SimpleAugmentor(AugmentationConfig(
            rotation_degrees=15, blur_prob=0.3, noise_prob=0.3, 
            color_jitter_prob=0.5, occlusion_prob=0.25, jpeg_prob=0.25, solarize_prob=0.15))
    raise ValueError(f"Unknown augmentation preset: {name}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CTCPlateDataset(Dataset):
    """Dataset for CTC training."""
    
    def __init__(
        self,
        records: Sequence[LicensePlateRecord],
        encoder: CharacterEncoder,
        target_height: int = 64,
        target_width: int = 256,
        augment_fn: Optional[Callable] = None,
        predict_province: bool = False,
        province_format: str = "code",
    ):
        self.records = list(records)
        self.encoder = encoder
        self.target_height = target_height
        self.target_width = target_width
        self.augment_fn = augment_fn
        self.predict_province = predict_province
        self.province_format = province_format
        
    def __len__(self) -> int:
        return len(self.records)
    
    def _build_label(self, record: LicensePlateRecord) -> str:
        """Build label text with or without province."""
        text = record.plate_text.strip()
        
        if self.predict_province and record.province_code:
            province_token = get_province_token(record.province_code, self.province_format)
            text = f"{text} {PROVINCE_SEP} {province_token}"
        
        return text
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        
        # Load and preprocess image
        image = load_plate_crop(record)
        
        # Apply augmentation
        if self.augment_fn is not None:
            image = self.augment_fn(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize((self.target_width, self.target_height))
        
        # Convert to tensor and normalize
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
        
        # Normalize using Simple normalization
        img_tensor = (img_tensor - 0.5) / 0.5
        
        # Encode label
        label_text = self._build_label(record)
        label_indices = self.encoder.encode(label_text)
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        return {
            "image": img_tensor,
            "label": label_tensor,
            "label_length": torch.tensor(len(label_indices), dtype=torch.long),
            "text": label_text,  # For debugging
        }


def ctc_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for CTC batches."""
    images = torch.stack([item["image"] for item in batch])
    
    # Pad labels to same length
    labels = [item["label"] for item in batch]
    label_lengths = torch.stack([item["label_length"] for item in batch])
    max_label_len = max(len(label) for label in labels)
    
    padded_labels = torch.zeros(len(labels), max_label_len, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label
    
    return {
        "images": images,
        "labels": padded_labels,
        "label_lengths": label_lengths,
    }


# ---------------------------------------------------------------------------
# CTC Model Architecture
# ---------------------------------------------------------------------------

class CTCModel(nn.Module):
    """CTC-based OCR model with CNN backbone + LSTM + FC."""
    
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet18",
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.3,
        image_height: int = 64,
        image_width: int = 256,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.rnn_hidden = rnn_hidden
        
        # CNN Backbone
        self.backbone = self._build_backbone(backbone)
        
        # Get feature dimension from backbone using actual input size
        dummy_input = torch.randn(1, 3, image_height, image_width)
        with torch.no_grad():
            dummy_out = self.backbone(dummy_input)
        
        # Feature dimensions: (batch, channels, height, width)
        _, c, h, w = dummy_out.shape
        self.feature_channels = c
        self.feature_height = h
        self.feature_width = w
        
        # Sequence dimension will be width, and we flatten height*channels
        self.rnn_input_size = c * h
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0,
            batch_first=True,
        )
        
        # Fully connected layer for classification
        self.fc = nn.Linear(rnn_hidden * 2, num_classes)  # *2 for bidirectional
        
        self.dropout = nn.Dropout(dropout)
        
    def _build_backbone(self, backbone: str) -> nn.Module:
        """Build CNN backbone for feature extraction."""
        if backbone == "resnet18":
            if models is None:
                raise ImportError("torchvision is required for ResNet backbones")
            model = models.resnet18(pretrained=True)
            # Remove avgpool and fc
            layers = list(model.children())[:-2]
            return nn.Sequential(*layers)
        
        elif backbone == "resnet34":
            if models is None:
                raise ImportError("torchvision is required for ResNet backbones")
            model = models.resnet34(pretrained=True)
            layers = list(model.children())[:-2]
            return nn.Sequential(*layers)
        
        elif backbone.startswith("efficientnet"):
            if timm is None:
                raise ImportError("timm is required for EfficientNet backbones. Install with `pip install timm`")
            model = timm.create_model(backbone, pretrained=True, features_only=True)
            return model
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (batch, 3, height, width)
        
        Returns:
            Log probabilities (batch, seq_len, num_classes)
        """
        # Extract features
        features = self.backbone(x)  # (batch, channels, h, w)
        
        # Reshape to sequence: (batch, width, channels*height)
        b, c, h, w = features.shape
        features = features.permute(0, 3, 1, 2)  # (batch, w, c, h)
        features = features.reshape(b, w, c * h)  # (batch, w, c*h)
        
        # Apply dropout
        features = self.dropout(features)
        
        # LSTM
        rnn_out, _ = self.rnn(features)  # (batch, w, rnn_hidden*2)
        
        # Apply dropout
        rnn_out = self.dropout(rnn_out)
        
        # Fully connected
        logits = self.fc(rnn_out)  # (batch, w, num_classes)
        
        # Log softmax for CTC loss
        log_probs = F.log_softmax(logits, dim=2)
        
        return log_probs


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def character_error_rate(prediction: str, reference: str) -> float:
    """Calculate Character Error Rate."""
    prediction = prediction or ""
    reference = reference or ""
    if not reference and not prediction:
        return 0.0
    if not reference:
        return float(len(prediction))
    if Levenshtein:
        dist = Levenshtein.distance(prediction, reference)
    else:
        dist = _levenshtein_distance(prediction, reference)
    return dist / max(1, len(reference))


def _levenshtein_distance(a: str, b: str) -> int:
    """Compute Levenshtein distance."""
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


def evaluate(
    model: CTCModel,
    dataloader: DataLoader,
    encoder: CharacterEncoder,
    device: torch.device,
    return_examples: bool = False,
    num_examples: int = 10,
) -> Dict[str, Any]:
    """Evaluate model on validation/test set."""
    model.eval()
    
    total_cer = 0.0
    exact_matches = 0
    total_samples = 0
    
    # Store examples for inspection
    examples = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["images"].to(device)
            labels = batch["labels"]
            label_lengths = batch["label_lengths"]
            
            # Forward pass
            log_probs = model(images)  # (batch, seq_len, num_classes)
            
            # Decode predictions (greedy decoding)
            _, predictions = torch.max(log_probs, dim=2)  # (batch, seq_len)
            
            # Decode to text
            pred_texts = encoder.decode_batch(predictions.cpu())
            
            # Decode ground truth
            gt_texts = []
            for i in range(len(labels)):
                length = label_lengths[i].item()
                gt_indices = labels[i, :length].tolist()
                gt_text = encoder.decode(gt_indices, remove_duplicates=False)
                gt_texts.append(gt_text)
            
            # Calculate metrics and collect examples
            for pred, gt in zip(pred_texts, gt_texts):
                cer = character_error_rate(pred, gt)
                total_cer += cer
                is_correct = pred.strip() == gt.strip()
                if is_correct:
                    exact_matches += 1
                total_samples += 1
                
                # Collect examples if requested
                if return_examples and len(examples) < num_examples:
                    examples.append({
                        "prediction": pred.strip(),
                        "ground_truth": gt.strip(),
                        "correct": is_correct,
                        "cer": cer,
                    })
    
    metrics = {
        "cer": total_cer / max(1, total_samples),
        "exact_match": exact_matches / max(1, total_samples),
    }
    
    if return_examples:
        metrics["examples"] = examples
    
    return metrics


def train_epoch(
    model: CTCModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        labels = batch["labels"].to(device)
        label_lengths = batch["label_lengths"].to(device)
        
        # Forward pass
        log_probs = model(images)  # (batch, seq_len, num_classes)
        
        # CTC loss expects (seq_len, batch, num_classes)
        log_probs = log_probs.permute(1, 0, 2)
        
        # Input lengths (all sequences have same length from CNN output)
        batch_size = log_probs.size(1)
        input_lengths = torch.full((batch_size,), log_probs.size(0), dtype=torch.long)
        
        # Compute loss
        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Progress bar
        progress = (batch_idx + 1) / total_batches
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        
        avg_loss = total_loss / num_batches
        
        # Update progress every batch
        print(f"\rEpoch {epoch} [{bar}] {batch_idx + 1}/{total_batches} | Loss: {avg_loss:.4f}", end="", flush=True)
        
        # Detailed log every 10 batches
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"\nEpoch {epoch}, Batch {batch_idx + 1}/{total_batches}, Current Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")
    
    print()  # New line after progress bar
    return total_loss / max(1, num_batches)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CTC-based Thai License Plate OCR")
    
    # Data
    parser.add_argument("--csv", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--data-root", type=str, action="append", required=True, help="Root directory containing images")
    parser.add_argument("--output-dir", type=str, default="outputs/ctc_model", help="Output directory")
    
    # Splits
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    
    # Model
    parser.add_argument("--backbone", type=str, default="resnet18", 
                       choices=["resnet18", "resnet34", "efficientnet_b0", "efficientnet_b1"],
                       help="CNN backbone for feature extraction. ResNet18 (~11M params, fastest), "
                            "ResNet34 (~21M params, more accurate), EfficientNet-B0 (~5M params, balanced), "
                            "EfficientNet-B1 (~7M params, most accurate but slower)")
    parser.add_argument("--rnn-hidden", type=int, default=256, help="LSTM hidden size")
    parser.add_argument("--rnn-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3)
    
    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-train-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--augment", type=str, default="light", 
                       choices=["none", "light", "medium", "heavy"])
    
    # Image size
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--image-width", type=int, default=384)
    
    # Province prediction
    parser.add_argument("--predict-province", action="store_true", help="Include province in labels")
    parser.add_argument("--province-format", type=str, default="code", 
                       choices=["code", "name"], help="Province token format")
    
    # Misc
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda)")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Load records
    logger.info(f"Loading records from {args.csv}")
    records = load_records(
        args.csv,
        data_roots=args.data_root,
        require_validate=True,
        drop_missing_images=True,
    )
    
    if not records:
        raise RuntimeError("No records loaded. Check CSV path and --data-root values.")
    
    logger.info(f"Loaded {len(records)} records")
    
    # Create stratified splits
    logger.info(f"Creating splits (train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio})")
    splits = stratified_split(
        records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    
    # Create character encoder
    encoder = CharacterEncoder(
        include_province=args.predict_province,
        province_format=args.province_format,
    )
    logger.info(f"Character encoder: {len(encoder)} classes")
    logger.info(f"Characters: {encoder.chars[:20]}...")
    
    # Build augmentation
    augment_fn = build_augmentor(args.augment)
    
    # Create datasets
    train_dataset = CTCPlateDataset(
        splits["train"],
        encoder,
        target_height=args.image_height,
        target_width=args.image_width,
        augment_fn=augment_fn,
        predict_province=args.predict_province,
        province_format=args.province_format,
    )
    
    val_dataset = CTCPlateDataset(
        splits["val"],
        encoder,
        target_height=args.image_height,
        target_width=args.image_width,
        augment_fn=None,
        predict_province=args.predict_province,
        province_format=args.province_format,
    )
    
    test_dataset = CTCPlateDataset(
        splits["test"],
        encoder,
        target_height=args.image_height,
        target_width=args.image_width,
        augment_fn=None,
        predict_province=args.predict_province,
        province_format=args.province_format,
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ctc_collate_fn,
        pin_memory=True if device.type == "cuda" else False,
    )
    
    # Create model
    logger.info(f"Building model with backbone: {args.backbone}")
    model = CTCModel(
        num_classes=len(encoder),
        backbone=args.backbone,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CTCLoss(blank=encoder.blank_idx, zero_infinity=True)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save encoder
    encoder_path = output_dir / "encoder.json"
    with open(encoder_path, "w", encoding="utf-8") as f:
        json.dump({
            "chars": encoder.chars,
            "char_to_idx": encoder.char_to_idx,
            "include_province": encoder.include_province,
            "province_format": encoder.province_format,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved encoder to {encoder_path}")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_cer = float("inf")
    
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_cer = checkpoint.get("best_cer", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}, best CER: {best_cer:.4f}")
    
    # Training loop
    logger.info("Starting training")
    
    for epoch in range(start_epoch, args.num_train_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1)
        logger.info(f"Training loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, encoder, device)
        logger.info(f"Validation - CER: {val_metrics['cer']:.4f}, Exact Match: {val_metrics['exact_match']:.4f}")
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_cer": val_metrics["cer"],
            "val_exact_match": val_metrics["exact_match"],
            "best_cer": best_cer,
        }
        
        # Save latest
        torch.save(checkpoint, output_dir / "checkpoint_latest.pt")
        
        # Save best
        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
            logger.info(f"✓ New best model saved (CER: {best_cer:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1}.pt")
    
    # Final evaluation on test set
    logger.info("\n" + "="*60)
    logger.info("Final evaluation on test set")
    logger.info("="*60)
    
    # Load best model
    best_checkpoint = torch.load(output_dir / "checkpoint_best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    
    # Evaluate with examples
    test_metrics = evaluate(model, test_loader, encoder, device, return_examples=True, num_examples=20)
    logger.info(f"Test - CER: {test_metrics['cer']:.4f}, Exact Match: {test_metrics['exact_match']:.4f}")
    
    # Display prediction examples
    logger.info("\n" + "="*60)
    logger.info("Prediction Examples (First 20 samples)")
    logger.info("="*60)
    
    for i, example in enumerate(test_metrics.get("examples", []), 1):
        status = "✓" if example["correct"] else "✗"
        logger.info(f"\n{status} Example {i}:")
        logger.info(f"  Predicted:    {example['prediction']}")
        logger.info(f"  Ground Truth: {example['ground_truth']}")
        logger.info(f"  CER: {example['cer']:.4f}")
    
    # Save test metrics (without examples to keep file small)
    metrics_to_save = {
        "cer": test_metrics["cer"],
        "exact_match": test_metrics["exact_match"],
        "examples": test_metrics.get("examples", []),
    }
    
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\nTraining complete! Models saved to {output_dir}")


if __name__ == "__main__":
    main()
