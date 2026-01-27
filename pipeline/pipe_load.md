# Pipeline Documentation: pipe_load.ipynb

## ภาพรวม (Overview)

`pipe_load.ipynb` เป็น pipeline แบบ end-to-end สำหรับการอ่านป้ายทะเบียนรถไทย โดยแบ่งเป็น 2 ขั้นตอนหลัก:

1. **Plate Detection** - ตรวจจับตำแหน่งป้ายทะเบียนในภาพ
2. **Plate Splitting** - แยกส่วนประกอบของป้ายทะเบียน (ตัวอักษร/ตัวเลข + จังหวัด)
3. **OCR + Province Classification** - อ่านข้อความและจำแนกจังหวัด

---

## โครงสร้าง Pipeline

```
Input Image
    ↓
[Plate Detector] (YOLOv11s)
    ↓
Plate Crop
    ↓
[Plate Splitter] (YOLOv11n)
    ↓
├─→ [OCR Model] (CRNN-CTC) → Plate Text
└─→ [Province Classifier] (MobileNetV3/EfficientNet) → Province Name
```

---

## Models & Weights

### 1. Plate Detector

- **Architecture**: YOLOv11s
- **Weight**: `weights/plate_detector_best.pt`
- **Input Size**: 1280px
- **Output**: Bounding boxes ของป้ายทะเบียนทั้งหมดในภาพ
- **Hyperparameters**:
  - `conf=0.25` - confidence threshold
  - `iou=0.7` - NMS IoU threshold

### 2. Plate Splitter

- **Architecture**: YOLOv11n
- **Weight**: `weights/plate_splitter_best.pt`
- **Input Size**: 640px
- **Output**: 2 classes
  - `class 0`: license_text (ตัวอักษร/ตัวเลข)
  - `class 1`: province (จังหวัด)
- **Hyperparameters**:
  - `conf=0.25`
  - `iou=0.6`

### 3. Province Classifier

- **Architecture**: `mobilenetv3_small_100` (new) หรือ `tf_efficientnet_b0` (old)
- **Weight**: `weights/province_classifier_best_new_model.pt`
- **Input Size**: 32×128 (H×W)
- **Output**: Top-3 provinces + confidence scores
- **Preprocessing**:
  ```python
  Resize(32, 128)
  ToTensor()
  Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  ```

### 4. OCR Model (CTC)

- **Architecture**: CRNN (CNN + BiLSTM + CTC)
  - CNN: Conv2d layers (64→128→256→512 channels)
  - RNN: 2-layer BiLSTM (256 hidden units)
  - Classifier: Linear layer
- **Weight**: `weights/upper_ctc_special_best.pt`
- **Input Size**: 32×128 (H×W), grayscale
- **Output**: Decoded text sequence
- **Preprocessing**:
  ```python
  Resize(32, 128)
  ToTensor()
  Normalize((0.5,), (0.5,))
  ```
- **Decoding**: Greedy CTC decoding (ตัด blank token และ repeats)

---

## วิธีการทำงาน (Workflow)

### Cell 1: Pipeline Execution (Single Image)

1. **โหลด models ทั้งหมด** (ครั้งเดียวตอนเริ่ม)
   - Plate detector, splitter, province classifier, OCR model
2. **อ่านภาพ input**

   ```python
   IMG_PATH = Path(r"D:\CodingD\ALPR\data\TEST_IMG\...")
   frame = load_bgr(IMG_PATH)
   ```

3. **Stage 1: Plate Detection**
   - ใช้ YOLO plate detector หา bounding boxes
   - เลือก plate ที่มี confidence สูงสุด (ถ้ามีหลายอัน)

4. **Stage 2: Plate Splitting**
   - Crop ป้ายทะเบียนที่ detect ได้
   - ใช้ splitter model แยกส่วน license_text และ province

5. **Stage 3a: Province Classification**
   - Crop region ของจังหวัด (class 1)
   - Run province classifier → ได้ top-3 predictions
   - แสดง top-1 พร้อม confidence

6. **Stage 3b: OCR**
   - Crop region ของตัวอักษร/ตัวเลข (class 0)
   - Run CTC OCR model → ได้ text prediction

7. **Visualization**
   - แสดงผล 2 ขั้นตอน:
     - Step 1: ภาพเต็มพร้อม plate bounding boxes
     - Step 2: Plate crops พร้อม split regions + predictions

8. **Timing**
   - วัดเวลาแต่ละขั้นตอน (plate, split, province, OCR)
   - คำนวณ average และ total inference time

---

### Cell 2: Evaluation (Batch)

**จุดประสงค์**: วัดประสิทธิภาพของ pipeline บน test set จริง

#### Input Data

- **Images**: `data/test_images/` (โครงสร้างตามที่ระบุใน labels.csv)
- **Labels**: `data/test_images/labels.csv`
  - Columns: `plate`, `province_description`, `image_name_gray`
  - ใช้ `image_name_gray` match กับไฟล์ใน folder

#### Metrics

##### 1. Detection Rates

- **Plate detection rate**: % ของภาพที่ detect ป้ายทะเบียนได้
- **Text box found rate**: % ของ plate ที่แยก license_text ได้
- **Province box found rate**: % ของ plate ที่แยก province ได้

##### 2. Accuracy

- **Plate text exact match**: Exact string match (case-sensitive)
- **Province exact match**: Top-1 province ตรงกับ ground truth
- **CER (Character Error Rate)**: Levenshtein distance / ความยาวของ GT
  ```
  CER = edit_distance(gt, pred) / len(gt)
  ```
- **WER (Word Error Rate)**: Levenshtein distance ระดับคำ
  ```
  WER = edit_distance(gt_words, pred_words) / len(gt_words)
  ```

##### 3. Timing

- **Per-stage latency**: mean และ p95 (95th percentile) ของแต่ละ model
  - Plate detector
  - Splitter
  - Province classifier
  - OCR
- **End-to-end latency**: รวมทุกขั้นตอน (รวม preprocessing)
- **FPS**: Frames per second = 1 / mean(end-to-end time)

##### 4. Confusion Matrix (Province)

- แสดง true label vs predicted label
- หา top confusions (คู่จังหวัดที่สับสนกันบ่อย)

#### Evaluation Logic

```python
for each image in test set:
    1. Detect plate → ถ้าไม่เจอ → record as miss
    2. ถ้าเจอหลาย plates → เลือก confidence สูงสุด
    3. Split plate → แยก text/province regions
    4. Run OCR on text region → compare กับ GT
    5. Run classifier on province region → compare กับ GT
    6. Compute CER, WER, accuracy
    7. Record timing for each stage
```

---

## การใช้งาน (Usage)

### 1. ติดตั้ง Dependencies

```python
# Core
ultralytics  # YOLOv11
torch
torchvision
timm  # Province classifier

# Utilities
opencv-python
pillow
matplotlib
pandas
numpy
```

### 2. เตรียม Weights

วาง weights files ใน `D:\CodingD\ALPR\weights\`:

- `plate_detector_best.pt`
- `plate_splitter_best.pt`
- `province_classifier_best_new_model.pt`
- `upper_ctc_special_best.pt` (หรือ `upper_ctc_best.pt` สำหรับป้ายปกติ)

### 3. รัน Pipeline (Single Image)

1. เปลี่ยน `IMG_PATH` ในเซลล์แรก
2. Run cell → ได้ผลลัพธ์ทันที:
   - Visualizations (2 steps)
   - Predictions (province + text)
   - Timing breakdown

### 4. รัน Evaluation

1. เตรียม test set:
   - วางรูปใน `data/test_images/`
   - สร้าง `labels.csv` ตามรูปแบบ:
     ```csv
     plate,province_description,image_name_gray
     กว1234,กรุงเทพมหานคร,001.jpg
     ```
2. Set `MAX_SAMPLES` (optional) สำหรับ quick test
3. Run cell → ได้ metrics ครบ + confusion matrix

---

## สิ่งที่ควรทราบ (Important Notes)

### 1. Thai Font Rendering

- ใช้ **Tahoma** font บน Windows สำหรับแสดงผลภาษาไทยใน matplotlib
- ถ้าไม่มี Tahoma → fallback ไป default font (อาจแสดงผิด)

### 2. Greedy CTC Decoding

- OCR ใช้ greedy decoding (เลือก character ที่ prob สูงสุดแต่ละ timestep)
- ตัด blank token (idx=0) และ consecutive repeats
- ไม่ได้ใช้ beam search → เร็วแต่อาจได้ผลแม่นน้อยกว่า

### 3. Province Classifier

- Return top-3 predictions แต่ evaluate แค่ top-1
- Confidence threshold ไม่ได้ใช้ → เอา top-1 เสมอ
- ถ้า province box ไม่เจอ → mark เป็น `<MISS>`

### 4. Multi-Plate Handling

- ถ้ามีหลาย plates ใน 1 ภาพ → เลือก confidence สูงสุด
- Evaluation: ประเมินแค่ plate แรก (อาจไม่ตรงกับ GT ถ้า GT เป็น plate อื่น)

### 5. Performance Tips

- **GPU**: ใช้ CUDA ถ้ามี → เร็วกว่า CPU มาก
- **Batch Inference**: Pipeline ปัจจุบันทำทีละรูป → อาจปรับให้ batch ได้
- **Image Size**:
  - Plate detector: imgsz=1280 → แม่นแต่ช้า, ลดเป็น 640 ได้ถ้าต้องการความเร็ว
  - Splitter: imgsz=640 → เหมาะสมแล้ว

---

## Failure Cases & Troubleshooting

### 1. No Plate Detected

- **สาเหตุ**: ภาพมืด, มุมกล้องแปลก, ป้ายเล็กเกินไป
- **แก้**: ลด confidence threshold หรือ เพิ่ม imgsz

### 2. Splitter แยกผิด

- **สาเหตุ**: ป้ายบางแบบ (ทะเบียนเก่า, ป้ายพิเศษ) ไม่ตรงกับ training data
- **แก้**: Augment training data หรือ retrain splitter

### 3. OCR ผิด

- **สาเหตุ**: ตัวอักษรเบลอ, มุมเอียง, แสงสะท้อน
- **แก้**: Preprocessing (denoise, deskew) หรือ retrain OCR model

### 4. Province ผิด

- **สาเหตุ**: จังหวัดบางอันมี logo คล้ายกัน
- **แก้**: ดู confusion matrix → augment data สำหรับคู่ที่สับสน

### 5. Slow Inference

- **สาเหตุ**: CPU inference, large image size
- **แก้**: ใช้ GPU, ลด imgsz, ใช้ model เล็กกว่า (YOLOv11n แทน v11s)

---

## Thesis-Ready Metrics (Suggested)

สำหรับเขียนวิทยานิพนธ์หรือรายงาน:

1. **Detection Performance**
   - Plate detection rate (%)
   - Text/Province split success rate (%)
2. **Recognition Accuracy**
   - OCR exact match accuracy (%)
   - CER (Character Error Rate)
   - WER (Word Error Rate)
   - Province classification accuracy (%)
3. **Speed**
   - Per-stage latency (ms): mean ± std, p95
   - End-to-end latency (ms)
   - Throughput (FPS)
4. **Error Analysis**
   - Failure breakdown: no plate / no text / no province
   - Province confusion matrix
   - Top-N confused province pairs
5. **Ablation Studies** (ถ้าทำ)
   - อิทธิพลของ confidence threshold
   - อิทธิพลของ image size
   - เปรียบเทียบ province classifier architectures (MobileNet vs EfficientNet)
   - เปรียบเทียบ OCR models (CTC vs Attention)

---

## Future Improvements

### 1. Beam Search Decoding (OCR)

- แทน greedy decoding ด้วย beam search → แม่นขึ้น
- Trade-off: ช้าขึ้นเล็กน้อย

### 2. Ensemble Province Classifier

- ใช้หลาย models vote กัน → ลด confusion

### 3. Post-Processing Rules

- ตรวจสอบรูปแบบป้ายทะเบียน (regex)
- แก้ characters ที่อ่านผิดบ่อย (O↔0, I↔1)

### 4. End-to-End Multi-Task Model

- ออกแบบ model เดียวทำ detection + split + OCR + province → เร็วกว่า pipeline

### 5. Real-Time Optimization

- Model quantization (FP16, INT8)
- TensorRT / ONNX Runtime
- Batch inference สำหรับ video

---

## References

- **YOLOv11**: https://github.com/ultralytics/ultralytics
- **timm**: https://github.com/huggingface/pytorch-image-models
- **CTC Loss**: Graves et al., "Connectionist Temporal Classification"
- **CRNN**: Shi et al., "An End-to-End Trainable Neural Network for Image-based Sequence Recognition"

---

## Contact & Support

หากพบปัญหาหรือต้องการปรับปรุง pipeline:

1. ตรวจสอบ weights files ว่าโหลดถูกต้อง
2. ดู error logs ใน Jupyter output
3. ทดสอบด้วยรูปง่ายๆ ก่อน (ป้ายชัด, แสงดี)
4. ปรับ hyperparameters ตามความเหมาะสม

**Happy License Plate Reading! 🚗📸**
