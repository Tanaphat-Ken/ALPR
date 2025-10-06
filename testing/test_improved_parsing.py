#!/usr/bin/env python3
"""Test the improved province parsing function."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.train_trocr import split_prediction_and_province

def test_parsing():
    """Test the improved parsing function with real examples."""
    
    test_cases = [
        '5กฆ921 President President',
        '8กฐ1674milimilincing',
        '9กข4267ขบ President',
        '2ขย967miliซอง',
        '3ฒย4299 Presidentประจวบคีรีขันธ์',
        '3กฆ9999miliฒ',
        '3ขย912ภรรยาของเขาmili',
        '1กษ1684milimili',
        '2ฒย4226 President President',
        '2ฒย6299 Presidentขบ',
        'กท2311 <prov> <TH-47>',  # Normal case
        '2ขย967 TH-10',  # Without <prov> but with TH code
    ]
    
    print("=== Testing Improved Province Parsing ===")
    
    for i, prediction in enumerate(test_cases, 1):
        plate, province = split_prediction_and_province(prediction)
        print(f"\nTest {i}:")
        print(f"  Input:    '{prediction}'")
        print(f"  Plate:    '{plate}'")
        print(f"  Province: '{province}'")

if __name__ == "__main__":
    test_parsing()