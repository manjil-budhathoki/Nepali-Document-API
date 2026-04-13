import logging
import cv2
import numpy as np
from typing import Tuple, Optional

from src.config import (
    YOLO_ROUTER_MODEL,
    YOLO_CITIZENSHIP_MODEL,
    OCR_LANG,
    OCR_MODEL_NAME,
    OCR_DEVICE
)


logger = logging.getLogger(__name__)

"""
Model cache ( singleton pattern) to avoid reloading models on every request. we will keep it in the memory so they only load once when the server starts.
"""
_models = {
    "router": None,
    "detector": None,
    "paddle": None
}

def load_models():
    """Warms up all AI models into memory at server startup."""
    logger.info("Loading AI models into memory...")

    try:
        from ultralytics import YOLO
        _models["router"] = YOLO(YOLO_ROUTER_MODEL)
        _models["detector"] = YOLO(YOLO_CITIZENSHIP_MODEL)
        logger.info("YOLO models loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading YOLO models: {e}")
    
    try:
        from paddleocr import PaddleOCR

        _models["paddle"] = PaddleOCR(
            text_recognition_model_name=OCR_MODEL_NAME,  # This uses "devanagari_PP-OCRv5_mobile_rec" from config
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            device=OCR_DEVICE  # This uses "cpu" from config
        )
        logger.info("✅ PaddleOCR loaded successfully with custom devanagari model.")

    except Exception as e:
        logger.error(f"❌ Failed to load PaddleOCR: {e}")
        import traceback
        traceback.print_exc()


# Document Classification
def get_document_type(image: np.ndarray) -> Optional[str]:
    """Detects if document is 'citizenship', 'demat' or 'unknown' using the router model."""
    if _models["router"] is None:
        logger.error("Router model not loaded.")
        raise RuntimeError("Router model not loaded.")
    
    results = _models["router"].predict(image, conf=0.5,verbose=False)

    detected_labels = set()
    for r in results:
        for box in r.boxes:
            label = _models["router"].names[int(box.cls[0])].lower()
            detected_labels.add(label)
    
    if any("demat" in label for label in detected_labels):
        return "demat"
    if any(kw in label for label in detected_labels for kw in ["citizenshipid", "citizenship"]):
        return "citizenship"
    
    return "unknown"

# Region detection and Cropping
def get_citizenship_crop(image:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the main text block, crops it IN MEMORY, and returns (cropped_image, visual_image_with_boxes).
    """
    if _models["detector"] is None:
        logger.error("Detector model not loaded.")
        raise RuntimeError("Detector model not loaded.")

    results = _models["detector"].predict(image, conf=0.4, verbose=False)
    result = results[0]
    
    crop_image = image  # Default to full image if detection fails
    visual_img = result.plot() # image with bounding boxes drawn

    for box in result.boxes:
        label = _models["detector"].names[int(box.cls[0])]
        if label == "text_block_primary":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # IN-MEMORY Crop (slicing the numpy array directly)
            crop_image = image[y1:y2, x1:x2]
            break
    
    return crop_image, visual_img

# OCR Extraction
def extract_text(image: np.ndarray, backend: str = "paddle") -> str:
    """
    Extracts text from a numpy image array.
    """


    if backend =="tesseract":
        import pytesseract

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return pytesseract.image_to_string(gray)

    elif backend == "paddle":

        if _models["paddle"] is None:
            raise RuntimeError("Paddle OCR Not Loaded")
        
        # pass the numpy array directly to the Paddle
        result = _models["paddle"].ocr(image)

        # parse Paddle's Nested list output
        extracted_texts = []
        if result and result[0]:
            for line in result[0]:
                # line format: [[box points], ("text", confidence)]
                text = line[1][0] 
                extracted_texts.append(text.strip())
                
        return " ".join(extracted_texts)
    
    else:
        raise ValueError(f"Unknown OCR backend: {backend}")