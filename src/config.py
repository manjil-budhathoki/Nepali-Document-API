"""
This file contains all the hardcoded paths and settings for the application.
"""
import os

# Model paths
YOLO_ROUTER_MODEL = os.getenv("YOLO_ROTER_MODEL", "models/yolo/best.pt")
YOLO_CITIZENSHIP_MODEL = os.getenv("YOLO_CITIZENSHIP_MODEL", "models/yolo/citizenship.pt")

# OCR settings
OCR_LANG = "ne"
OCR_MODEL_NAME = "devanagari_PP-OCRv5_mobile_rec"
OCR_DEVICE = "cpu" # Change to "cuda" if you have a compatible GPU and the necessary drivers installed

# Matching Thresholds
FUZZY_MATCH_THRESHOLD_HIGH = 80
FUZZY_MATCH_THRESHOLD_LOW = 50