"""
Synthetic Thai License Plate Generator with Bounding Box Labels
Generates synthetic Thai license plates with:
- Thai characters and numbers in license plate format
- Province names (77 provinces)
- Homography transformation for distortion
- Data augmentation (translation, rotation, scaling)
- Bicubic interpolation for sharpness
- Bounding box annotations for license number (top) and province (bottom)
"""

import os
import random
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from pathlib import Path
import argparse
from typing import Tuple, List, Dict, Optional


# Thai provinces (77 provinces)
THAI_PROVINCES = [
    "กระบี่", "กรุงเทพมหานคร", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร",
    "ขอนแก่น", "จันทบุรี", "จำปาศักดิ์", "ฉะเชิงเทรา", "ชลบุรี",
    "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่",
    "ตรัง", "ตราด", "ตาก", "นครนายก", "นครปฐม",
    "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี",
    "นราธิวาส", "น่าน", "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี",
    "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา", "พังงา",
    "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์",
    "แพร่", "พะเยา", "ภูเก็ต", "มหาสารคาม", "มุกดาหาร",
    "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง",
    "ระยอง", "ราชบุรี", "ลพบุรี", "ลำปาง", "ลำพูน",
    "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล",
    "สมุทรปราการ", "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี",
    "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี", "สุรินทร์",
    "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อุดรธานี", "อุทัยธานี",
    "อุตรดิตถ์", "อุบลราชธานี", "อำนาจเจริญ"
]

# Thai consonants commonly used in license plates
THAI_CONSONANTS = [
    "ก", "ข", "ค", "ง", "จ", "ฉ", "ช", "ซ", "ฌ", "ญ",
    "ฎ", "ฏ", "ฐ", "ฑ", "ฒ", "ณ", "ด", "ต", "ถ", "ท",
    "ธ", "น", "บ", "ป", "ผ", "ฝ", "พ", "ฟ", "ภ", "ม",
    "ย", "ร", "ล", "ว", "ศ", "ษ", "ส", "ห", "ฬ", "อ", "ฮ"
]

# Common Thai words for synthetic data (sample dictionary)
THAI_WORDS = [
    "กรุงเทพ", "ไทย", "รถ", "บ้าน", "เมือง", "คน", "น้ำ", "ฟ้า",
    "ดิน", "ไฟ", "ลม", "ป่า", "ภูเขา", "ทะเล", "แม่น้ำ", "ดอกไม้",
    "ต้นไม้", "สวน", "ถนน", "สะพาน", "วัด", "โรงเรียน", "โรงพยาบาล",
    "ตลาด", "ร้านค้า", "อาหาร", "ข้าว", "น้ำตาล", "เกลือ", "พริก",
    "หมู", "ไก่", "เป็ด", "ปลา", "กุ้ง", "ปู", "ผัก", "ผลไม้"
]

# Thai words for special/vanity license plates (ป้ายทะเบียนพิเศษ)
# These words have meaning and often include vowels above/below characters
THAI_SPECIAL_WORDS = [
    # คำมงคล (Auspicious words)
    "รวย", "โชค", "ดี", "เฮง", "เจริญ", "สุข", "ศรี", "มั่ง", "มี", "โภค",
    "ทรัพย์", "สมบูรณ์", "เจริญ", "รุ่ง", "เรือง", "วัฒนา", "พัฒนา",
    # คำที่น่าสนใจ (Interesting words)
    "หล่อ", "สวย", "เท่", "เจ๋ง", "เฟี้ยว", "ว้าว", "โอ้โห",
    # คำทั่วไป (Common words)
    "รัก", "หวัง", "ฝัน", "ใจ", "คิด", "เชื่อ", "ไว้", "ได้", "เอา",
    "ไป", "มา", "อยู่", "เป็น", "ทำ", "ให้", "กับ", "แล้ว", "เพื่อ",
    # คำสั้น (Short words)
    "ฟ้า", "ดิน", "น้ำ", "ไฟ", "ลม", "ดอก", "ใบ", "เมฆ", "ฝน", "แดด"
]


class ThaiLicensePlateGenerator:
    """Generate synthetic Thai license plates with bounding box labels"""
    
    def __init__(
        self,
        output_dir: str = "synthetic_plates",
        image_width: int = 340,
        image_height: int = 150,
        font_path: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        
        self.image_width = image_width
        self.image_height = image_height
        
        # Try to load Thai font
        self.font_path = font_path or self._find_thai_font()
        self.load_fonts()
        
    def _find_thai_font(self) -> str:
        """Find available Thai font on the system"""
        possible_fonts = [
            "C:/Users/PC/AppData/Local/Microsoft/Windows/Fonts/Sarun's ThangLuang.ttf",  # Default Windows Thai font
            "C:/Windows/Fonts/THSarabunNew.ttf",
            "C:/Windows/Fonts/THSarabun Bold.ttf",
            "/usr/share/fonts/truetype/tlwg/Sarabun-Bold.ttf",
            "/usr/share/fonts/truetype/tlwg/Loma-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf",
            "/System/Library/Fonts/Thonburi.ttc",  # macOS
        ]
        
        for font in possible_fonts:
            if os.path.exists(font):
                print(f"Using font: {font}")
                return font
        
        print("Warning: No Thai font found, using default")
        return None
    
    def load_fonts(self):
        """Load fonts for license plate text"""
        try:
            if self.font_path:
                # Thai characters (letters) - larger size
                self.font_thai_chars = ImageFont.truetype(self.font_path, 50)
                # Numbers - smaller size to balance with Thai characters
                self.font_numbers = ImageFont.truetype(self.font_path, 42)
                # Province name (bottom) - medium size for better visibility
                self.font_province = ImageFont.truetype(self.font_path, 30)
            else:
                self.font_thai_chars = ImageFont.load_default()
                self.font_numbers = ImageFont.load_default()
                self.font_province = ImageFont.load_default()
        except Exception as e:
            print(f"Error loading font: {e}")
            self.font_thai_chars = ImageFont.load_default()
            self.font_numbers = ImageFont.load_default()
            self.font_province = ImageFont.load_default()
    
    def generate_plate_text(self, special_plate_ratio: float = 0.0) -> Tuple[Dict[str, str], str]:
        """
        Generate Thai license plate text in 3 parts
        Returns: (plate_parts, province)
        
        plate_parts format:
        {
            'prefix_number': '1' or '',  # Optional prefix number (for some formats)
            'thai_chars': 'กข' or 'หล่อ',  # Thai characters or word
            'suffix_number': '1234' or '999'  # Suffix numbers
        }
        
        Format: 
        - Standard: [optional: 1 digit] + 2-3 Thai consonants + 4 digits (e.g., "1กข 1234" or "กขค 5678")
        - Special: Thai word + 3-4 digits (e.g., "หล่อ 999" or "รวย 1234")
        
        Args:
            special_plate_ratio: Probability of generating special/vanity plates (0.0 to 1.0)
        """
        # Decide whether to generate special plate
        use_special = random.random() < special_plate_ratio
        
        if use_special:
            # Generate special/vanity license plate
            word = random.choice(THAI_SPECIAL_WORDS)
            # Special plates often have 3 digits, but can have 4
            num_digits = random.choice([3, 3, 3, 4])  # 75% chance for 3 digits
            suffix_digits = ''.join([str(random.randint(0, 9)) for _ in range(num_digits)])
            
            plate_parts = {
                'prefix_number': '',
                'thai_chars': word,
                'suffix_number': suffix_digits
            }
        else:
            # Generate standard license number
            # Some plates have prefix number (e.g., "1กข 1234"), some don't (e.g., "กข 1234")
            has_prefix = random.random() < 0.3  # 30% chance for prefix number
            prefix = str(random.randint(1, 9)) if has_prefix else ''
            
            # 2-3 Thai consonants
            num_consonants = random.choice([2, 3])
            consonants = ''.join(random.choices(THAI_CONSONANTS, k=num_consonants))
            
            # 4 digits suffix
            suffix_digits = ''.join([str(random.randint(0, 9)) for _ in range(4)])
            
            plate_parts = {
                'prefix_number': prefix,
                'thai_chars': consonants,
                'suffix_number': suffix_digits
            }
        
        # Select random province
        province = random.choice(THAI_PROVINCES)
        
        return plate_parts, province
    
    def generate_background_color(self) -> Tuple[int, int, int]:
        """Generate random background color for license plate"""
        # Common Thai license plate colors (more realistic)
        colors = [
            (255, 255, 255),  # White (most common - personal vehicles)
            (255, 235, 100),  # Yellow (taxi)
            (255, 180, 90),   # Orange (motorcycle taxi)
            (120, 255, 120),  # Light green (government)
        ]
        
        # 80% chance for white (most common), 20% for others
        if random.random() < 0.8:
            return colors[0]
        else:
            return random.choice(colors[1:])
    
    def create_plate_image(
        self,
        plate_parts: Dict[str, str],
        province: str
    ) -> Tuple[Image.Image, str]:
        """
        Create license plate image with text in 3 parts
        Returns: (image, full_license_text)
        
        Args:
            plate_parts: Dictionary with 'prefix_number', 'thai_chars', 'suffix_number'
            province: Province name in Thai
        """
        # Create blank image with background color
        bg_color = self.generate_background_color()
        img = Image.new('RGB', (self.image_width, self.image_height), bg_color)
        draw = ImageDraw.Draw(img)

        # Add border
        border_color = (0, 0, 0)
        border_width = 3
        draw.rectangle(
            [0, 0, self.image_width-1, self.image_height-1],
            outline=border_color,
            width=border_width
        )

        # Text color (black on light background)
        text_color = (0, 0, 0)

        # Calculate dimensions for each part
        prefix_number = plate_parts['prefix_number']
        thai_chars = plate_parts['thai_chars']
        suffix_number = plate_parts['suffix_number']
        
        # Get bounding boxes for each part to calculate widths
        prefix_width = 0
        if prefix_number:
            try:
                bbox_prefix = draw.textbbox((0, 0), prefix_number, font=self.font_numbers, anchor='ls')
                prefix_width = bbox_prefix[2] - bbox_prefix[0]
            except:
                prefix_width = len(prefix_number) * 30
        
        try:
            bbox_thai = draw.textbbox((0, 0), thai_chars, font=self.font_thai_chars, anchor='ls')
            thai_width = bbox_thai[2] - bbox_thai[0]
        except:
            thai_width = len(thai_chars) * 35
        
        try:
            bbox_suffix = draw.textbbox((0, 0), suffix_number, font=self.font_numbers, anchor='ls')
            suffix_width = bbox_suffix[2] - bbox_suffix[0]
        except:
            suffix_width = len(suffix_number) * 30
        
        # Calculate spacing and total width
        space_prefix_thai = 3  # Smaller space between prefix number and Thai chars
        space_thai_suffix = 10  # Normal space between Thai chars and suffix number
        total_width = prefix_width + (space_prefix_thai if prefix_number else 0) + thai_width + space_thai_suffix + suffix_width
        
        # Starting x position (centered)
        start_x = (self.image_width - total_width) // 2
        
        # Use baseline anchor 'ls' (left-baseline) for consistent alignment
        # This ensures all characters align at the same baseline regardless of descenders (หาง)
        baseline_y = 70  # Baseline position (adjusted from top)
        
        # Draw prefix number (if exists)
        current_x = start_x
        if prefix_number:
            draw.text(
                (current_x, baseline_y),
                prefix_number,
                font=self.font_numbers,
                fill=text_color,
                anchor='ls'  # left-baseline anchor
            )
            current_x += prefix_width + space_prefix_thai
        
        # Draw Thai characters
        draw.text(
            (current_x, baseline_y),
            thai_chars,
            font=self.font_thai_chars,
            fill=text_color,
            anchor='ls'  # left-baseline anchor
        )
        current_x += thai_width + space_thai_suffix
        
        # Draw suffix number
        draw.text(
            (current_x, baseline_y),
            suffix_number,
            font=self.font_numbers,
            fill=text_color,
            anchor='ls'  # left-baseline anchor
        )

        # Draw province name (bottom, centered)
        try:
            bbox_province = draw.textbbox((0, 0), province, font=self.font_province, anchor='lt')
            province_width = bbox_province[2] - bbox_province[0]
            province_height = bbox_province[3] - bbox_province[1]
            province_offset_y = -bbox_province[1]
        except:
            province_width = len(province) * 15
            province_height = 25
            province_offset_y = 0

        province_x = (self.image_width - province_width) // 2
        province_y = self.image_height - province_height - 20

        draw.text(
            (province_x, province_y + province_offset_y),
            province,
            font=self.font_province,
            fill=text_color,
            anchor='lt'
        )

        # Create full license text for CSV output
        full_license = f"{prefix_number}{thai_chars} {suffix_number}".strip()

        return img, full_license
    
    def apply_homography(self, img: np.ndarray) -> np.ndarray:
        """
        Apply homography transformation to simulate perspective distortion
        """
        height, width = img.shape[:2]
        
        # Define random perspective transformation
        # Source points (corners of the image)
        src_points = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        
        # Destination points (with random distortion)
        distortion = random.uniform(0.05, 0.15)
        dst_points = np.float32([
            [random.uniform(0, width*distortion), random.uniform(0, height*distortion)],
            [width - random.uniform(0, width*distortion), random.uniform(0, height*distortion)],
            [width - random.uniform(0, width*distortion), height - random.uniform(0, height*distortion)],
            [random.uniform(0, width*distortion), height - random.uniform(0, height*distortion)]
        ])
        
        # Calculate homography matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        transformed = cv2.warpPerspective(img, matrix, (width, height), 
                                         flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_REPLICATE)
        
        return transformed
    
    def apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation: translation, rotation, scaling
        """
        height, width = img.shape[:2]
        
        # Random rotation (-15 to 15 degrees)
        angle = random.uniform(-15, 15)
        rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (width, height),
                            flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
        
        # Random scaling (0.8 to 1.2)
        scale = random.uniform(0.85, 1.15)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Crop or pad to original size
        if scale > 1.0:
            # Crop center
            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            img = img[start_y:start_y+height, start_x:start_x+width]
        else:
            # Pad with border color
            pad_x = (width - new_width) // 2
            pad_y = (height - new_height) // 2
            img = cv2.copyMakeBorder(img, pad_y, height-new_height-pad_y,
                                    pad_x, width-new_width-pad_x,
                                    cv2.BORDER_REPLICATE)
        
        # Random translation (-20 to 20 pixels)
        tx = random.randint(-20, 20)
        ty = random.randint(-10, 10)
        translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, translation_matrix, (width, height),
                            borderMode=cv2.BORDER_REPLICATE)
        
        return img
    
    def add_noise_and_blur(self, img: np.ndarray) -> np.ndarray:
        """Add random noise and blur to make image more realistic"""
        # Add Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, random.uniform(5, 15), img.shape)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Add blur
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # Adjust brightness
        if random.random() < 0.3:
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        return img
    
    def apply_bicubic_interpolation(self, img: np.ndarray, scale: float = 0.5) -> np.ndarray:
        """
        Apply bicubic interpolation to adjust image sharpness
        Downscale then upscale to simulate different quality levels
        """
        height, width = img.shape[:2]
        
        # Downscale
        small = cv2.resize(img, (int(width*scale), int(height*scale)),
                          interpolation=cv2.INTER_CUBIC)
        
        # Upscale back to original size
        img = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
        
        return img
    
    def generate_sample(
        self,
        index: int,
        apply_transforms: bool = True,
        special_plate_ratio: float = 0.0
    ) -> Dict:
        """
        Generate a single synthetic license plate sample for TrOCR training
        Returns: metadata dictionary matching train_trocr.py CSV format
        
        Note: Bounding boxes are NOT needed for TrOCR training as it reads the entire image.
        """
        # Generate text parts
        plate_parts, province = self.generate_plate_text(special_plate_ratio)
        
        # Create base image with 3-part rendering
        img_pil, full_license = self.create_plate_image(plate_parts, province)
        
        # Convert to numpy array
        img_np = np.array(img_pil)
        
        if apply_transforms:
            # Apply homography transformation
            if random.random() < 0.7:
                img_np = self.apply_homography(img_np)
            
            # Apply augmentation
            if random.random() < 0.8:
                img_np = self.apply_augmentation(img_np)
            
            # Apply bicubic interpolation (simulate different quality)
            if random.random() < 0.5:
                scale = random.uniform(0.4, 0.8)
                img_np = self.apply_bicubic_interpolation(img_np, scale)
            
            # Add noise and blur
            img_np = self.add_noise_and_blur(img_np)
        
        # Save image with relative path format matching training data
        image_filename = f"synthetic_{index:06d}.jpg"
        image_path = self.images_dir / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        # Map province name to province code (use TH-XX format)
        province_code = f"TH-{THAI_PROVINCES.index(province) + 1:02d}"
        
        # Create CSV row data matching train_trocr.py format
        # Only includes fields actually used by TrOCR training
        label_data = {
            "plate": full_license,
            "province_code": province_code,
            "province_description": province,
            "image_name_gray": f"images/{image_filename}",
        }
        
        return label_data
    
    def generate_dataset(
        self,
        num_samples: int = 1000,
        apply_transforms: bool = True,
        special_plate_ratio: float = 0.0
    ):
        """Generate a complete dataset of synthetic license plates"""
        print(f"Generating {num_samples} synthetic license plates...")
        print(f"Output directory: {self.output_dir}")
        if special_plate_ratio > 0:
            print(f"Special/vanity plate ratio: {special_plate_ratio*100:.0f}%")
        
        metadata = []
        
        for i in range(num_samples):
            try:
                label_data = self.generate_sample(i, apply_transforms, special_plate_ratio)
                metadata.append(label_data)
                
                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{num_samples} samples")
            
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save as CSV file matching training data format
        csv_path = self.output_dir / "synthetic_plates.csv"
        
        # Write CSV header and data
        import csv
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            # Define CSV columns matching train_trocr.py requirements
            # Note: Bounding boxes removed as TrOCR doesn't need them
            fieldnames = [
                'plate',
                'province_code',
                'province_description',
                'image_name_gray'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in metadata:
                writer.writerow(row)
        
        print(f"\nDataset generation complete!")
        print(f"- Images: {self.images_dir}")
        print(f"- CSV file: {csv_path}")
        print(f"- Total samples: {len(metadata)}")
        print(f"- Image size: {self.image_width}x{self.image_height}")
        print(f"\nYou can now use this CSV file for training with train_trocr.py")
        print(f"Example command:")
        print(f"  python train/train_trocr.py --csv {csv_path} --data-root {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic Thai license plates with bounding box labels"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="synthetic_plates",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=340,
        help="Image width (default: 340, matching real Thai license plates)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=150,
        help="Image height (default: 150, matching real Thai license plates)"
    )
    parser.add_argument(
        "--font-path",
        type=str,
        default=None,
        help="Path to Thai font file (.ttf)"
    )
    parser.add_argument(
        "--no-transforms",
        action="store_true",
        help="Disable transformations (for debugging)"
    )
    parser.add_argument(
        "--special-ratio",
        type=float,
        default=0.0,
        help="Ratio of special/vanity plates (0.0-1.0, default: 0.0)"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = ThaiLicensePlateGenerator(
        output_dir=args.output_dir,
        image_width=args.width,
        image_height=args.height,
        font_path=args.font_path
    )
    
    # Generate dataset
    generator.generate_dataset(
        num_samples=args.num_samples,
        apply_transforms=not args.no_transforms,
        special_plate_ratio=args.special_ratio
    )


if __name__ == "__main__":
    main()
