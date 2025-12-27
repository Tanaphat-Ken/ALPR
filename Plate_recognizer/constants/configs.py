import os 
import ast
from dotenv import load_dotenv

load_dotenv()

# Model Weights
CAR_LOCALIZER_WEIGHT="models/weights/yolov8n.pt"
PLATE_LOCALIZER_WEIGHT="models/weights/license_plate_detector.pt"
CRAFT_WEIGHT="models/weights/craft_mlt_25k.pth"
CRAFT_REFINER_WEIGHT="models/weights/craft_refiner_CTW1500.pth"
CHARACTOR_READER_WEIGHT="models/weights/charactor_reader.pth"

# CRAFT parameters
TEXT_THRESHOLD = 0.8
LINK_THRESHOLD = 0.4
LOW_TEXT = 0.7

ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/jpg"]

ALLOW_ORIGINS = ast.literal_eval(os.getenv("allow_origins", "[]"))
ALLOW_METHODS = ast.literal_eval(os.getenv("allow_methods", "[]"))
ALLOW_HEADERS = ast.literal_eval(os.getenv("allow_headers", "[]"))