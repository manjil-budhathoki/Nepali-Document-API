# --- 2. REGION DETECTION & CROPPING (Replaces object_detector.py) ---
def get_citizenship_crop(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the main text block, crops it IN MEMORY, and returns (cropped_image, visual_image_with_boxes).
    """
    if _models["detector"] is None:
        raise RuntimeError("Detector model not loaded")

    results = _models["detector"].predict(image, conf=0.4, verbose=False)
    result = results[0]
    
    crop_image = image  # Default to full image if detection fails
    visual_img = result.plot() # Image with drawn boxes for debugging
    
    for box in result.boxes:
        label = _models["detector"].names[int(box.cls[0])]
        if label == "text_block_primary":
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # IN-MEMORY CROP (Slicing the numpy array directly!)
            crop_image = image[y1:y2, x1:x2]
            break
            
    return crop_image, visual_img


# --- 3. OCR EXTRACTION (Replaces ocr_reader.py) ---
def extract_text(image: np.ndarray, backend: str = "paddle") -> str:
    """Extracts text from a numpy image array."""
    
    if backend == "tesseract":
        # Tesseract logic (Usually used for Demat)
        import pytesseract
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return pytesseract.image_to_string(gray).strip()
        
    elif backend == "paddle":
        # PaddleOCR logic (Used for Citizenship/Nepali)
        if _models["paddle"] is None:
            raise RuntimeError("PaddleOCR not loaded")
            
        # Pass the numpy array directly to Paddle (No temp files!)
        result = _models["paddle"].ocr(image, cls=True)
        
        # Parse Paddle's nested list output
        extracted_texts = []
        if result and result[0]:
            for line in result[0]:
                # line format: [[box points], ("text", confidence)]
                text = line[1][0] 
                extracted_texts.append(text.strip())
                
        return " ".join(extracted_texts)
    
    else:
        raise ValueError(f"Unknown OCR backend: {backend}")