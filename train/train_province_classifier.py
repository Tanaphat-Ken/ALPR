"""Train province classifier using full license plate images."""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import logging
import sys

# Add parent directory to path to import train modules
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("train_province")

# Import province mappings from existing module
from train.province_mapping import PROVINCE_CODE_TO_THAI_NAME

# Province code to ID mapping (Thai 2-letter codes)
PROVINCE_CODES = [
    "กท", "กพ", "กส", "ขก", "ขน", "ตร", "ชบ", "ชม", "ชย", "ตง",
    "นบ", "นน", "นฐ", "นพ", "นม", "นว", "นส", "บร", "บก", "ปน",
    "พย", "พล", "พช", "พท", "พน", "พม", "พร", "พะ", "พิ", "มห",
    "ยล", "ยส", "รบ", "ระ", "รย", "ลบ", "ลย", "ศก", "สน", "สข",
    "สพ", "สม", "สร", "สก", "สต", "สบ", "สส", "อท", "อน", "อบ",
    "อย", "ชร", "ตก", "ปต", "พง", "ลป", "สท", "หนอ", "กน", "นค",
    "ภก", "สห", "นฐ", "ลพ", "กย", "อด", "ชพ", "กจ", "ปข", "สง",
    "สค", "นศ", "มส", "ชล", "อต", "กฉ", "นก"
]

PROVINCE_CODE_TO_ID = {code: i for i, code in enumerate(PROVINCE_CODES)}
ID_TO_PROVINCE_CODE = {i: code for i, code in enumerate(PROVINCE_CODES)}

# Create ISO to Thai province code mapping from province names
# This extracts first 2 Thai consonants from province name as the code
def _create_iso_to_thai_mapping():
    """Auto-generate ISO to Thai province code mapping."""
    mapping = {}
    
    # Manual mapping for provinces where auto-extraction doesn't work well
    # or where standardized codes differ from name-based extraction
    manual_overrides = {
        "TH-10": "กท",  # กรุงเทพมหานคร -> กท (not กร)
        "TH-11": "สป",  # สมุทรปราการ -> สป (not สม)
        "TH-12": "นฐ",  # นนทบุรี -> นฐ (not นน)
        "TH-13": "ปท",  # ปทุมธานี -> ปท
        "TH-14": "อย",  # พระนครศรีอยุธยา -> อย (not พน)
        "TH-19": "สระ", # สระบุรี -> สระ (not สร - conflicts with สุรินทร์)
        "TH-74": "สม",  # สมุทรสาคร -> สม (not สส)
        "TH-75": "สส",  # สมุทรสงคราม -> สส
    }
    
    for iso_code, thai_name in PROVINCE_CODE_TO_THAI_NAME.items():
        if iso_code in manual_overrides:
            mapping[iso_code] = manual_overrides[iso_code]
        else:
            # Extract first 2 Thai consonants from province name
            consonants = [c for c in thai_name if '\u0e01' <= c <= '\u0e2e']  # Thai consonants range
            if len(consonants) >= 2:
                province_code = consonants[0] + consonants[1]
                mapping[iso_code] = province_code
    
    return mapping

ISO_TO_THAI_PROVINCE = _create_iso_to_thai_mapping()

class ProvinceDataset(Dataset):
    """Dataset for province classification using full plate images."""
    
    def __init__(
        self, 
        records: list,
        target_height: int = 128,
        target_width: int = 384,
        augment_fn = None
    ):
        self.records = records
        self.target_height = target_height
        self.target_width = target_width
        self.augment_fn = augment_fn
        
    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        from PIL import Image
        import numpy as np
        from train.data_utils import load_plate_crop
        
        record = self.records[idx]
        
        # Load full plate image
        image = load_plate_crop(record)
        
        # Apply augmentation
        if self.augment_fn:
            image = self.augment_fn(image)
        
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize((self.target_width, self.target_height))
        
        # Convert to tensor
        img_array = np.array(image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Normalize (simple normalization for all plate colors)
        img_tensor = (img_tensor - 0.5) / 0.5
        
        # Get province label
        province_code = record.province_code
        
        # Convert ISO format (TH-XX) to Thai province code if needed
        if province_code and province_code.startswith("TH-"):
            province_code = ISO_TO_THAI_PROVINCE.get(province_code, None)
        
        # Skip records with unknown province codes
        if not province_code or province_code not in PROVINCE_CODE_TO_ID:
            return None
        
        label = PROVINCE_CODE_TO_ID[province_code]
        
        return {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "province_code": province_code,
        }


class ProvinceClassifier(nn.Module):
    """ResNet-based classifier for Thai provinces (77 classes)."""
    
    def __init__(self, num_provinces: int = 77, backbone: str = "resnet18"):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
            num_features = 512
        elif backbone == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
            num_features = 512
        elif backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
            num_features = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_provinces),
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


def train_province_model(
    csv_path: str,
    data_root: list,
    output_dir: str = "outputs/province_classifier",
    backbone: str = "resnet18",
    batch_size: int = 64,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
):
    """Train province classifier."""
    
    from train.data_utils import load_records, stratified_split
    from train.train_ctc import build_augmentor
    import time
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Display device info
    print(f"\n{'='*70}")
    print(f"🖥️  Device Configuration")
    print(f"{'='*70}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ Using GPU: {gpu_name}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {gpu_memory:.2f} GB")
        
        # Auto-adjust batch size for Colab GPU
        if "Tesla T4" in gpu_name and batch_size < 128:
            print(f"   💡 Colab T4 detected, recommend batch_size=128")
        elif gpu_memory < 8 and batch_size > 64:
            batch_size = 64
            print(f"   ⚠️  Limited GPU memory, reducing batch_size to {batch_size}")
    else:
        print(f"⚠️  Using CPU (training will be slower)")
        print(f"   💡 For faster training, use Colab with GPU:")
        print(f"      Runtime → Change runtime type → GPU (T4)")
        if batch_size > 32:
            batch_size = 32
            print(f"   Reducing batch_size to {batch_size} for CPU")
    print(f"{'='*70}\n")
    
    # Load data
    logger.info(f"📁 Loading records from {csv_path}")
    records = load_records(csv_path, data_roots=data_root, require_validate=True)
    
    # Filter records with province
    records = [r for r in records if r.province_code]
    logger.info(f"✓ Found {len(records)} records with province codes")
    
    # Split data
    splits = stratified_split(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Create datasets
    augment_fn = build_augmentor("medium")
    train_dataset = ProvinceDataset(splits["train"], augment_fn=augment_fn)
    val_dataset = ProvinceDataset(splits["val"], augment_fn=None)
    test_dataset = ProvinceDataset(splits["test"], augment_fn=None)
    
    logger.info(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Custom collate function
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None
        images = torch.stack([item["image"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        return {"image": images, "label": labels}
    
    # Optimize DataLoader for Colab
    num_workers = 2 if torch.cuda.is_available() else 0  # Colab has limited CPU cores
    pin_memory = torch.cuda.is_available()
    
    logger.info(f"✓ DataLoader: batch_size={batch_size}, num_workers={num_workers}, pin_memory={pin_memory}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn, 
        pin_memory=pin_memory, persistent_workers=num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory
    )
    
    # Create model
    print(f"\n🏗️  Building {backbone} model...")
    model = ProvinceClassifier(num_provinces=len(PROVINCE_CODES), backbone=backbone)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model: {backbone}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_acc = 0.0
    epoch_times = []
    
    print(f"\n{'='*70}")
    print(f"🚀 Starting Training")
    print(f"{'='*70}\n")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        total_batches = len(train_loader)
        print(f"📚 Epoch {epoch+1}/{num_epochs} - Training")
        
        for batch_idx, batch in enumerate(train_loader):
            if batch is None:
                continue
            
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
            
            # Progress bar (update every 5 batches or at end)
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                progress = (batch_idx + 1) / total_batches
                bar_length = 40
                filled = int(bar_length * progress)
                bar = "█" * filled + "░" * (bar_length - filled)
                
                avg_loss = train_loss / (batch_idx + 1)
                curr_acc = train_correct / train_total if train_total > 0 else 0
                
                print(f"\r[{bar}] {batch_idx+1}/{total_batches} | "
                      f"Loss: {avg_loss:.4f} | Acc: {curr_acc:.4f}", end="", flush=True)
        
        print()  # New line
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        print(f"🔍 Validating...", end="", flush=True)
        
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                
                images = batch["image"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        print(f" Done!")
        
        # Calculate epoch time and ETA
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = num_epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_hours = eta_seconds / 3600
        eta_minutes = (eta_seconds % 3600) / 60
        
        # Display summary
        print(f"\n📊 Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"   Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"   Train Acc:  {train_acc:.4f} ({train_correct}/{train_total})")
        print(f"   Val Acc:    {val_acc:.4f} ({val_correct}/{val_total})")
        print(f"   LR:         {optimizer.param_groups[0]['lr']:.6f}")
        print(f"   Time:       {epoch_time:.1f}s")
        if remaining_epochs > 0:
            if eta_hours >= 1:
                print(f"   ETA:        {int(eta_hours)}h {int(eta_minutes)}m ({remaining_epochs} epochs)")
            else:
                print(f"   ETA:        {int(eta_seconds/60)}m {int(eta_seconds%60)}s ({remaining_epochs} epochs)")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "train_acc": train_acc,
                "province_codes": PROVINCE_CODES,
            }, output_path / "best_model.pt")
            print(f"   ✅ New best model! (Val Acc: {val_acc:.4f})")
        
        # Save periodic checkpoint (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "optimizer_state_dict": optimizer.state_dict(),
            }, output_path / f"checkpoint_epoch_{epoch+1}.pt")
            print(f"   💾 Checkpoint saved (epoch {epoch+1})")
        
        scheduler.step()
        print()  # Spacing between epochs
    
    # Test
    print(f"\n{'='*70}")
    print(f"Final Evaluation on Test Set")
    print(f"{'='*70}")
    
    model.load_state_dict(torch.load(output_path / "best_model.pt")["model_state_dict"])
    model.eval()
    test_correct = 0
    test_total = 0
    
    # Track per-province accuracy
    province_correct = {}
    province_total = {}
    
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:  # Skip empty batches
                continue
            
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            # Overall accuracy
            test_correct += (predicted == labels).sum().item()
            test_total += labels.size(0)
            
            # Per-province accuracy
            for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                prov_code = ID_TO_PROVINCE_CODE[label]
                if prov_code not in province_total:
                    province_correct[prov_code] = 0
                    province_total[prov_code] = 0
                province_total[prov_code] += 1
                if pred == label:
                    province_correct[prov_code] += 1
    
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    print(f"\n🎯 Test Results:")
    print(f"   Overall Accuracy: {test_acc:.4f} ({test_correct}/{test_total})")
    print(f"\n📈 Top 10 Provinces by Sample Count:")
    
    # Sort provinces by sample count
    sorted_provinces = sorted(province_total.items(), key=lambda x: x[1], reverse=True)[:10]
    for prov_code, total in sorted_provinces:
        correct = province_correct.get(prov_code, 0)
        acc = correct / total if total > 0 else 0
        print(f"   {prov_code}: {acc:.4f} ({correct}/{total})")
    
    print(f"\n{'='*70}")
    print(f"Training Complete! Best Val Acc: {best_acc:.4f}")
    print(f"Models saved to: {output_path}")
    print(f"{'='*70}\n")
    
    # Save config
    with open(output_path / "config.json", "w") as f:
        json.dump({
            "num_provinces": len(PROVINCE_CODES),
            "province_codes": PROVINCE_CODES,
            "backbone": backbone,
            "test_accuracy": test_acc,
        }, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--data-root", action="append", required=True)
    parser.add_argument("--output-dir", default="outputs/province_classifier")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    
    args = parser.parse_args()
    
    train_province_model(
        csv_path=args.csv,
        data_root=args.data_root,
        output_dir=args.output_dir,
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
    )