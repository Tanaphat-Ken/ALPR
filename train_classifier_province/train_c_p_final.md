# Province Classifier Training Documentation

## ภาพรวม (Overview)

`train_classifier_province_final.ipynb` เป็น notebook สำหรับเทรน **Province Classifier** ที่จำแนกจังหวัด 78 คลาส (77 จังหวัด + เบตง) จากภาพ lower crop (ส่วนจังหวัดของป้ายทะเบียน) ขนาด 32×128 pixels

**Use Case**: ใช้ใน inference pipeline (RTSP, real-time) เพื่อจำแนกจังหวัดหลังจากที่ splitter แยก province region ออกมาจากป้ายทะเบียน

---

## Model Architectures

### Recommended Models

1. **tf_efficientnet_b0** (ค่าเริ่มต้นแนะนำ)
   - Accuracy สูง (~95-97%)
   - Speed: ปานกลาง (~5-10ms/inference)
   - Model size: ~17MB
   - Parameters: ~5M

2. **mobilenetv3_small_100** (สำหรับ real-time)
   - Accuracy: ดี (~93-95%)
   - Speed: เร็วมาก (~2-5ms/inference)
   - Model size: ~6MB
   - Parameters: ~2M

### Trade-offs

| Model                 | Accuracy   | Speed      | Size   | Use Case                    |
| --------------------- | ---------- | ---------- | ------ | --------------------------- |
| tf_efficientnet_b0    | ⭐⭐⭐⭐⭐ | ⭐⭐⭐     | Medium | Production (accuracy-first) |
| mobilenetv3_small_100 | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐ | Small  | Real-time / Edge devices    |

---

## Dataset Structure

### Expected Format

```
lower_train/
  ├── labels.csv
  └── data/
      ├── img001.jpg
      ├── img002.jpg
      └── ...

lower_test/
  ├── labels.csv
  └── data/
      └── ...

lower_test_synthetic/
  ├── labels.csv
  └── data/
      └── ...
```

### labels.csv Format

**Required Columns**:

- `filename`: ชื่อไฟล์รูป (จะถูกต่อกับ `data/` folder)
- `province_description` หรือ `label`: ชื่อจังหวัด (เลือกอันใดอันหนึ่ง)

**Example**:

```csv
filename,province_description
img001.jpg,กรุงเทพมหานคร
img002.jpg,เชียงใหม่
img003.jpg,ภูเก็ต
```

### Data Filtering

- **Excluded Labels**: `ไม่พบข้อมูล` (unknown/missing province) จะถูกตัดออกโดยอัตโนมัติ
- **File Existence Check**: ตรวจสอบว่าไฟล์มีอยู่จริงก่อน load
- **Label Mapping**: สร้าง `label2idx` และ `idx2label` สำหรับ 78 classes

---

## Training Pipeline

### Cell-by-Cell Breakdown

#### Cell 1: Markdown - Title

- อธิบาย dataset และ suggested models

#### Cell 2: Dataset Paths & Unzip (Local + Colab)

- **Local Mode**: ค้นหา `data/` folder จาก repo root โดยอัตโนมัติ
- **Colab Mode**:
  - Mount Google Drive
  - Unzip `lower_train.zip`, `lower_test.zip`, `lower_test_synthetic.zip` จาก MyDrive
  - แกะไฟล์ไปที่ `/content/datasets/`
- Validate ว่ามี `labels.csv` ครบทุก split

#### Cell 3: Install Dependencies

```python
timm==0.9.16          # Model architectures
pandas==2.2.2         # Data handling
scikit-learn==1.5.2   # Train/val split
tqdm==4.66.4          # Progress bars
pillow                # Image loading
torchvision           # Transforms
```

#### Cell 4: Imports + Random Seed

- Set seed = 42 สำหรับ reproducibility
- Enable `torch.backends.cudnn.benchmark` สำหรับ speed

#### Cell 5: Config (CFG dataclass)

**Key Parameters**:

| Parameter         | Value (Local)         | Value (Colab)         | Description              |
| ----------------- | --------------------- | --------------------- | ------------------------ |
| `model_name`      | mobilenetv3_small_100 | mobilenetv3_small_100 | Model architecture       |
| `num_classes`     | 78                    | 78                    | จังหวัด 77 + เบตง        |
| `img_h` × `img_w` | 32×128                | 32×128                | Input size               |
| `epochs`          | 20                    | 20                    | Training epochs          |
| `batch_size`      | 64                    | 256                   | Batch size               |
| `lr`              | 3e-4                  | 3e-4                  | Learning rate            |
| `weight_decay`    | 1e-4                  | 1e-4                  | L2 regularization        |
| `val_ratio`       | 0.10                  | 0.10                  | Validation split         |
| `amp`             | True                  | True                  | Mixed precision training |
| `num_workers`     | 0                     | 2                     | DataLoader workers       |
| `debug_run`       | True                  | False                 | Debug mode (limit data)  |

**Debug Mode** (for quick laptop testing):

- `debug_train_per_class=150`: จำกัด 150 รูปต่อคลาส
- `debug_eval_rows=3000`: จำกัด eval ที่ 3000 รูป
- `max_train_batches=50`: จำกัด 50 batches/epoch
- `max_val_batches=20`: จำกัด 20 batches/epoch

**Save Directory**:

- Local: `./artifacts/alpr_province_classifier/`
- Colab: `/content/drive/MyDrive/ALPR_Project/alpr_province_classifier_final/`

#### Cell 6: Load Labels + Build Label Map

```python
# Read labels from CSV
train_df = read_labels(TRAIN_DIR)

# Build class list (sorted alphabetically)
classes = sorted(train_df['label'].unique())

# Create mappings
label2idx = {c: i for i, c in enumerate(classes)}
idx2label = {i: c for c, i in label2idx.items()}

# Save label_map.json (for inference)
```

**Filtering**:

1. เลือก column: `province_description` > `label`
2. Strip whitespace
3. ตัด `excluded_labels` (เช่น "ไม่พบข้อมูล")
4. ตรวจสอบ file exists
5. Debug mode: sample `debug_train_per_class` รูปต่อคลาส

#### Cell 7: Stratified Train/Val Split

- **Method**: `StratifiedShuffleSplit` (sklearn)
- **Ratio**: 90% train / 10% val
- **Stratified**: รักษาสัดส่วน class เท่าเดิมใน train/val
- **Seed**: 42 (reproducible)

```python
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
train_idx, val_idx = next(sss.split(train_df['path'], train_df['y']))
```

#### Cell 8: Data Augmentation + DataLoader

**Train Transforms**:

```python
T.Resize((32, 128))
T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)  # p=0.7
T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))                       # p=0.2
T.RandomAffine(degrees=2, translate=(0.02,0.05), scale=(0.95,1.05), shear=1)
T.ToTensor()
T.RandomErasing(p=0.3, scale=(0.02, 0.15))
T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

**Val Transforms**:

```python
T.Resize((32, 128))
T.ToTensor()
T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
```

**DataLoader Settings**:

- Train: `shuffle=True`, `drop_last=True`
- Val: `shuffle=False`
- `pin_memory=True` (GPU speedup)
- `persistent_workers=True` (Colab only) - workers รออยู่ตลอด ไม่ต้องสร้างใหม่ทุก epoch

#### Cell 9: Model + Loss + Optimizer + Scheduler

**Model**:

```python
model = timm.create_model(
    cfg.model_name,
    pretrained=True,          # ImageNet weights
    num_classes=78,
    in_chans=3,
    drop_rate=0.3             # Dropout
)
```

**Loss Function**: Weighted Cross-Entropy

```python
# คำนวณ class weights จากความถี่
counts = train_df['y'].value_counts().sort_index()
weights = total_samples / counts
weights = weights / weights.mean()  # normalize
criterion = nn.CrossEntropyLoss(weight=weights)
```

→ ช่วยแก้ class imbalance (จังหวัดบางอันมีรูปน้อย)

**Optimizer**: AdamW

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4  # L2 regularization
)
```

**Scheduler**: Cosine Annealing

```python
scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_steps  # ค่อยๆ ลด lr จาก 3e-4 → 0
)
```

**Mixed Precision Training** (AMP):

```python
scaler = torch.cuda.amp.GradScaler(enabled=True)
```

→ เร็วขึ้น ~30-50%, ประหยัด memory

#### Cell 10: Training Loop

**Features**:

1. **Early Stopping**: หยุดถ้า val_acc ไม่ขึ้น 3 epochs ติดต่อกัน
2. **Best Checkpoint**: บันทึก model ที่ val_acc สูงสุด
3. **Progress Bar**: แสดง loss/acc/lr แบบ real-time
4. **History Logging**: บันทึกทุก epoch ลง `history.csv`

**Saved Checkpoint** (`best.pt`):

```python
{
    'model_name': 'mobilenetv3_small_100',
    'num_classes': 78,
    'state_dict': model.state_dict(),
    'label2idx': {...},
    'idx2label': {...},
    'epoch': 15,
    'val_acc': 0.9543
}
```

**Debug Mode**: จำกัด batches ตามที่ตั้งไว้

```python
max_batches = cfg.max_train_batches if cfg.debug_run else None
```

#### Cell 11: Evaluation (Test Sets)

**Test on 2 Datasets**:

1. **lower_test**: Real test data (ภาพจริง)
2. **lower_test_synthetic**: Synthetic test data (ภาพสังเคราะห์)

**Metrics**:

- Top-1 Accuracy (exact match)

**Debug Mode**: sample `cfg.debug_eval_rows` รูป

**Output Example**:

```
[lower_test] n=12543 acc=0.9543
[lower_test_synthetic] n=8932 acc=0.9312
```

#### Cell 12: Inference Helper

**Function**: `predict_province(image_path, topk=5)`

**Usage**:

```python
results = predict_province('province_crop.jpg', topk=3)
# [('กรุงเทพมหานคร', 0.9523),
#  ('นนทบุรี', 0.0312),
#  ('ปทุมธานี', 0.0089)]
```

**เหมาะสำหรับ**:

- RTSP pipeline
- Real-time inference
- Web API (Flask/FastAPI)

---

## Training Strategy

### 1. Transfer Learning

- เริ่มจาก ImageNet pretrained weights
- Fine-tune ทุก layers (ไม่ freeze)

### 2. Regularization

- **Dropout**: 0.3 (ก่อน classifier layer)
- **Weight Decay**: 1e-4 (L2 regularization)
- **Data Augmentation**: ColorJitter, Blur, Affine, Erasing
- **Early Stopping**: patience=3 epochs

### 3. Class Imbalance Handling

- **Weighted Loss**: ให้ weight สูงกับ class ที่มีรูปน้อย
- **Stratified Split**: รักษาสัดส่วน class ใน train/val

### 4. Learning Rate Schedule

- **Initial LR**: 3e-4
- **Scheduler**: Cosine Annealing (ลดค่อยๆ ไป 0)
- **Steps**: อัปเดตทุก batch (not epoch)

### 5. Mixed Precision Training

- ใช้ FP16 สำหรับ forward/backward
- ใช้ FP32 สำหรับ optimizer step
- เร็วขึ้น ~30-50% บน GPU

---

## Metrics & Evaluation

### Training Metrics (per epoch)

| Metric       | Description                |
| ------------ | -------------------------- |
| `train_loss` | Cross-entropy loss (train) |
| `train_acc`  | Top-1 accuracy (train)     |
| `val_loss`   | Cross-entropy loss (val)   |
| `val_acc`    | Top-1 accuracy (val)       |
| `secs`       | Time per epoch             |

### Test Metrics

| Metric             | Description                           |
| ------------------ | ------------------------------------- |
| Top-1 Accuracy     | % ของรูปที่ predict ถูก (exact match) |
| Per-Class Accuracy | Accuracy แยกตามจังหวัด (optional)     |

### Expected Performance

| Model                 | Real Test Acc | Synthetic Test Acc | Inference Time |
| --------------------- | ------------- | ------------------ | -------------- |
| tf_efficientnet_b0    | ~95-97%       | ~93-95%            | ~5-10ms        |
| mobilenetv3_small_100 | ~93-95%       | ~91-93%            | ~2-5ms         |

---

## File Outputs

### Saved Files (in `cfg.save_dir`)

1. **label_map.json**

   ```json
   {
     "label2idx": {"กระบี่": 0, "กรุงเทพมหานคร": 1, ...},
     "idx2label": {"0": "กระบี่", "1": "กรุงเทพมหานคร", ...}
   }
   ```

   → ใช้ตอน inference เพื่อ decode prediction

2. **best.pt**
   - Model checkpoint ที่ val_acc สูงสุด
   - ประกอบด้วย: `model_name`, `num_classes`, `state_dict`, `label2idx`, `idx2label`

3. **history.csv**
   ```csv
   epoch,train_loss,train_acc,val_loss,val_acc,secs
   1,1.2345,0.7543,0.9876,0.8234,45.2
   2,0.9123,0.8432,0.7654,0.8765,44.8
   ...
   ```
   → ใช้วิเคราะห์ training curve

---

## การใช้งาน (Usage)

### 1. เตรียม Environment

**Local (Laptop)**:

```bash
cd train_classifier_province/
pip install timm pandas scikit-learn tqdm pillow torchvision torch
jupyter notebook train_classifier_province_final.ipynb
```

**Colab**:

1. อัปโหลด notebook ไป Colab
2. เตรียม zip files ใน Google Drive:
   - `MyDrive/ALPRV2/lower_train.zip`
   - `MyDrive/ALPRV2/lower_test.zip`
   - `MyDrive/ALPRV2/lower_test_synthetic.zip`
3. Run ทุก cell ตามลำดับ

### 2. เลือก Model

แก้ใน Cell 5:

```python
cfg.model_name = 'tf_efficientnet_b0'  # แม่นขึ้น
# หรือ
cfg.model_name = 'mobilenetv3_small_100'  # เร็วขึ้น
```

### 3. Debug Mode (Quick Test)

```python
cfg.debug_run = True  # จำกัด data + batches
cfg.epochs = 3        # รันแค่ 3 epochs
```

### 4. Production Training

```python
cfg.debug_run = False  # ใช้ data เต็ม
cfg.epochs = 20
cfg.batch_size = 256   # ถ้ามี GPU ดี
```

### 5. Load Checkpoint (Inference)

```python
import torch, timm
from PIL import Image
import torchvision.transforms as T

# Load checkpoint
ckpt = torch.load('best.pt', map_location='cuda')
model = timm.create_model(
    ckpt['model_name'],
    pretrained=False,
    num_classes=ckpt['num_classes']
).cuda()
model.load_state_dict(ckpt['state_dict'])
model.eval()

# Transforms
tfm = T.Compose([
    T.Resize((32, 128)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Predict
img = Image.open('province_crop.jpg').convert('RGB')
x = tfm(img).unsqueeze(0).cuda()
logits = model(x)
probs = logits.softmax(1).squeeze(0)
top5_vals, top5_idxs = probs.topk(5)

idx2label = ckpt['idx2label']
for v, i in zip(top5_vals, top5_idxs):
    print(f"{idx2label[str(int(i))]}: {v.item()*100:.2f}%")
```

---

## สิ่งที่ควรทราบ (Important Notes)

### 1. Dataset Quality

- **Crop ต้องแม่นยำ**: ถ้า splitter ตัด province crop ผิด → classifier จะเรียนผิด
- **Class Balance**: บางจังหวัดมีรูปน้อย → ใช้ weighted loss ช่วย
- **Synthetic Data**: มี distribution ต่างจาก real data → test แยกต่างหาก

### 2. Model Selection

- **Accuracy-first**: `tf_efficientnet_b0`
- **Speed-first**: `mobilenetv3_small_100`
- **Custom**: ลอง `efficientnet_lite0`, `mobilenetv3_large_100`, `resnet18`

### 3. Hyperparameter Tuning

| Parameter     | Low  | High | Effect                                     |
| ------------- | ---- | ---- | ------------------------------------------ |
| Learning Rate | 1e-4 | 5e-4 | ต่ำ=ช้าแต่เสถียร, สูง=เร็วแต่อาจไม่ลู่เข้า |
| Batch Size    | 64   | 512  | เล็ก=ช้า, ใหญ่=เร็ว (ต้องมี GPU ดี)        |
| Weight Decay  | 1e-5 | 1e-3 | ต่ำ=อาจ overfit, สูง=อาจ underfit          |
| Dropout       | 0.1  | 0.5  | ต่ำ=อาจ overfit, สูง=อาจ underfit          |

### 4. Data Augmentation

- **ไม่ควรใช้มาก**: Flip, Rotation มากเกินไป อาจทำให้ logo จังหวัดเพี้ยน
- **แนะนำ**: ColorJitter, Blur, RandomErasing, Affine (เล็กน้อย)

### 5. Colab Tips

- **GPU Runtime**: เปลี่ยนจาก CPU → GPU (Runtime > Change runtime type)
- **Persistent Workers**: `cfg.persistent_workers=True` (เร็วขึ้นมาก)
- **Drive Mount**: ทุกครั้งที่ Colab disconnect ต้อง mount ใหม่

---

## Troubleshooting

### 1. Out of Memory (OOM)

**สาเหตุ**: batch_size ใหญ่เกินไป  
**แก้**: ลด `cfg.batch_size` (64 → 32 → 16)

### 2. Val Accuracy ไม่ขึ้น

**สาเหตุ**: Overfitting, LR สูงเกิน, หรือ data ไม่ดี  
**แก้**:

- เพิ่ม dropout, weight_decay
- ลด learning rate
- เพิ่ม augmentation
- ตรวจสอบ train/val split (ควรมี class ครบทุกอัน)

### 3. Train/Val Accuracy ห่างกันมาก

**สาเหตุ**: Overfitting  
**แก้**:

- เพิ่ม regularization (dropout, weight_decay)
- เพิ่ม augmentation
- ลด model complexity (เปลี่ยนเป็น mobilenet)

### 4. Inference ช้า

**สาเหตุ**: Model ใหญ่เกิน  
**แก้**:

- เปลี่ยนเป็น `mobilenetv3_small_100`
- ใช้ model quantization (FP16, INT8)
- ใช้ TensorRT / ONNX Runtime

### 5. Confusion Between Similar Provinces

**สาเหตุ**: Logo ของบางจังหวัดคล้ายกัน (เช่น กทม vs นนทบุรี)  
**แก้**:

- Collect data เพิ่มสำหรับคู่ที่สับสน
- ใช้ focal loss แทน cross-entropy
- Ensemble หลาย models

### 6. FileNotFoundError: labels.csv

**สาเหตุ**: Path ไม่ถูก หรือ zip ไม่ครบ  
**แก้**:

- ตรวจสอบ `TRAIN_DIR`, `TEST_DIR` ว่า print ออกมาถูกไหม
- ตรวจสอบว่า zip file มี `labels.csv` อยู่ใน root หรือ subfolder

---

## Advanced Topics

### 1. Class-wise Analysis

```python
import numpy as np
from sklearn.metrics import classification_report

# Collect predictions
y_true = []
y_pred = []
for x, y in tqdm(val_loader):
    logits = model(x.to(device))
    y_true.extend(y.numpy())
    y_pred.extend(logits.argmax(1).cpu().numpy())

# Generate report
report = classification_report(
    y_true, y_pred,
    target_names=[idx2label[i] for i in range(len(idx2label))],
    digits=4
)
print(report)
```

### 2. Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, annot=False, cmap='Blues',
            xticklabels=[idx2label[i] for i in range(78)],
            yticklabels=[idx2label[i] for i in range(78)])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
```

### 3. Ensemble Multiple Models

```python
# Train 3 models
models = [
    ('efficientnet_b0', 'best_effb0.pt'),
    ('mobilenetv3_small', 'best_mobv3.pt'),
    ('resnet18', 'best_resnet18.pt')
]

# Load all
loaded = []
for name, path in models:
    ckpt = torch.load(path)
    m = timm.create_model(name, num_classes=78).cuda()
    m.load_state_dict(ckpt['state_dict'])
    m.eval()
    loaded.append(m)

# Ensemble prediction (average logits)
def ensemble_predict(img_tensor):
    logits = [m(img_tensor) for m in loaded]
    avg_logits = torch.stack(logits).mean(0)
    return avg_logits.argmax(1)
```

### 4. Model Quantization (Speed Up)

```python
# Convert to FP16
model.half()
x = x.half()  # input must also be FP16

# Or use torch.quantization (INT8)
import torch.quantization
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(model, inplace=True)
# Calibrate with sample data
torch.quantization.convert(model, inplace=True)
```

---

## Integration with Pipeline

### ใน pipe_load.ipynb

```python
# Load province classifier
province_ckpt = torch.load('best.pt')
province_model = timm.create_model(
    province_ckpt['model_name'],
    pretrained=False,
    num_classes=province_ckpt['num_classes']
).to(device)
province_model.load_state_dict(province_ckpt['state_dict'])
province_model.eval()

idx2label = province_ckpt['idx2label']

# Transforms
province_tfm = T.Compose([
    T.Resize((32, 128)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# Predict
@torch.no_grad()
def predict_province_from_bgr(bgr_crop):
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    x = province_tfm(img).unsqueeze(0).to(device)
    logits = province_model(x)
    probs = F.softmax(logits, 1).squeeze(0)
    vals, idxs = torch.topk(probs, k=3)
    return [(idx2label[str(int(i))], float(v)) for v, i in zip(vals, idxs)]
```

---

## Thesis-Ready Metrics (Suggested)

1. **Model Comparison**
   - Accuracy (train/val/test) for each architecture
   - Inference time (ms/image)
   - Model size (MB)

2. **Training Curves**
   - Loss vs epoch (train/val)
   - Accuracy vs epoch (train/val)

3. **Confusion Matrix**
   - 78×78 heatmap
   - Top-10 confused pairs

4. **Per-Class Accuracy**
   - Precision/Recall/F1 per province
   - Identify hard classes

5. **Ablation Studies**
   - Effect of data augmentation
   - Effect of weighted loss
   - Effect of transfer learning (pretrained vs random init)

6. **Robustness**
   - Accuracy on real vs synthetic data
   - Accuracy by image quality (blur, noise, lighting)

---

## Future Improvements

### 1. Attention Mechanism

- เพิ่ม CBAM (Convolutional Block Attention Module)
- ช่วยให้ model focus ที่ logo จังหวัด

### 2. Metric Learning

- ใช้ ArcFace / CosFace loss แทน Cross-Entropy
- ทำให้ feature space compact กว่า

### 3. Knowledge Distillation

- Train model ใหญ่ (teacher) แล้ว distill ลง model เล็ก (student)
- ได้ accuracy สูงแต่ inference เร็ว

### 4. Active Learning

- ให้ model เลือกรูปที่ uncertain → นำไป label เพิ่ม
- ประหยัด labeling cost

### 5. Multi-Task Learning

- Train ร่วมกับ OCR task
- Share encoder → ประหยัด computation

---

## References

- **timm**: https://github.com/huggingface/pytorch-image-models
- **EfficientNet**: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs" (ICML 2019)
- **MobileNetV3**: Howard et al., "Searching for MobileNetV3" (ICCV 2019)
- **Data Augmentation**: "A survey on Image Data Augmentation for Deep Learning" (2019)
- **Class Imbalance**: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)

---

## Contact & Support

หากพบปัญหาหรือต้องการปรับปรุง:

1. ตรวจสอบ dataset paths และ labels.csv format
2. ลอง debug_run mode ก่อน (รันเร็ว)
3. ดู history.csv และ training curves
4. ปรับ hyperparameters ตามตาราง
5. ทดสอบ inference ด้วย Cell 12

**Happy Province Classification Training! 🚗🏙️**
