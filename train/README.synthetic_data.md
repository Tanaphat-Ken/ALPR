# Synthetic Thai License Plate Generator (Latest Version)

This script generates synthetic Thai license plate images and CSV labels for OCR model training (e.g., TrOCR). It supports both standard and special/vanity plate formats, with realistic layout and font handling. The generator now renders the license plate in 3 distinct parts for better control over alignment and scaling.

## Key Features

- **3-Part Plate Rendering:**
  - Prefix number (optional, e.g., "1")
  - Thai characters/word (e.g., "กข" or "หล่อ")
  - Suffix number (e.g., "1234" or "999")
- **Baseline Alignment:** All text parts are aligned to the same baseline for perfect visual alignment, even with Thai letters that have descenders (หาง).
- **Customizable Spacing:** Prefix number and Thai characters are placed close together, mimicking real plates.
- **Province Text:** Province name is rendered clearly at the bottom, with adjustable font size.
- **Special/Vanity Plate Support:** Use `--special-ratio` to control the proportion of special plates.
- **Data Augmentation:** Perspective distortion, rotation, scaling, translation, noise, blur, and brightness adjustment (can be disabled for debugging).
- **Thai Font Support:** Specify a font via `--font-path` for best results.
- **TrOCR-Compatible CSV Output:** Only the fields required for TrOCR: `plate`, `province_code`, `province_description`, `image_name_gray`.

## Requirements

- Python 3.x
- [Pillow](https://python-pillow.org/)
- [OpenCV](https://opencv.org/)
- [numpy](https://numpy.org/)
- Thai font file (e.g., Sarun's ThangLuang.ttf, THSarabunNew.ttf)

Install dependencies:

```bash
pip install pillow opencv-python numpy
```

## Usage

Run the script from the project root or the `train` directory:

```bash
python train/synthetic_data.py [OPTIONS]
```

### Options

- `--output-dir` Output directory for generated data (default: `synthetic_plates`)
- `--num-samples` Number of samples to generate (default: 100)
- `--width` Image width in pixels (default: 340)
- `--height` Image height in pixels (default: 150)
- `--font-path` Path to a Thai font file (.ttf) (optional)
- `--no-transforms` Disable all image augmentations (for debugging)
- `--special-ratio` Ratio of special/vanity plates (0.0-1.0, default: 0.0)

### Example

Generate 1000 images with 30% special plates and a custom font:

```bash
python train/synthetic_data.py --num-samples 1000 --output-dir synthetic_plates_trocr --special-ratio 0.3 --font-path "C:/Users/PC/AppData/Local/Microsoft/Windows/Fonts/Sarun's ThangLuang.ttf"
```

## Output

- Images are saved in `[output-dir]/images/`
- CSV file is saved as `[output-dir]/synthetic_plates.csv`
- Each CSV row contains:
  - `plate`: License plate text (e.g., 1กข 1234, หล่อ 999)
  - `province_code`: Province code (e.g., TH-01)
  - `province_description`: Province name in Thai
  - `image_name_gray`: Relative path to the image file

## Notes

- Bounding box information is not included, as TrOCR requires only the full image and label.
- The script will auto-detect a Thai font if `--font-path` is not provided, but specifying a font is recommended for best results.
- The layout and alignment now closely match real Thai license plates, including spacing and baseline handling.
- You can use the generated CSV directly with your TrOCR training script.

## Example TrOCR Training Command

```bash
python train/train_trocr.py --csv synthetic_plates.csv --data-root synthetic_plates
```

---

**Author:** Tanaphat-Ken

For questions or improvements, please open an issue or contact the repository maintainer.
