import cv2
import base64
import numpy as np

from src.ml_engine import get_document_type, get_citizenship_crop, extract_text
from src.text_engine import (
    clean_and_repair_text,
    extract_demat_fields,
    verify_name,
    verify_id_number,
    verify_dob
)

def encode_image_base64(image:np.ndarray) -> str:
    """Convert a numpy array back to base64 for the fronted to display"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

def process_document(image:np.ndarray, user_data:dict) -> str:
    """Main pipeline:
        Classifies
        Extract
        verifies
    """

    # Classify document type
    doc_type = get_document_type(image)

    if not doc_type:
       return {"status": "error", "message": "Could not identify document type. Please upload a valid Demat or Citizenship document."}
    
    # Process based on type
    if doc_type == "citizenship":

        # Crop 
        crop_img, debug_img = get_citizenship_crop(image)

        # OCR
        raw_text = extract_text(crop_img, backend="paddle")
        clean_text = clean_and_repair_text(raw_text)

        # Text Verification
        verification = {
            "name": verify_name(user_data.get("name"), clean_text),
            "citizenship_number": verify_id_number(user_data.get("citizenship_number"), clean_text),
            "dob": verify_dob(user_data.get("dob"), clean_text)
        }

        return {
            "status": "success",
            "document_type": "citizenship",
            "extracted_text": clean_text,
            "verification": verification,
            "debug_images": {"yolo_boxes": encode_image_base64(debug_img)}
        }


    elif doc_type == "demat":

        # OCR Full page
        raw_text = extract_text(image, backend="tesseract")

        # Text Extraction
        extracted_data = extract_demat_fields(raw_text)

        # Verification
        verification = {}
        
        # Map the User's keys to the Extracted keys
        fields_to_check = [
            {"user_key": "name", "ext_key": "name"},
            {"user_key": "boid", "ext_key": "boid"},
            {"user_key": "dob", "ext_key": "dob"},
            {"user_key": "citizenship_number", "ext_key": "citizenship"},
            {"user_key": "contact_number", "ext_key": "contact"}, 
        ]
        
        for field in fields_to_check:
            u_key = field["user_key"]
            e_key = field["ext_key"]
            
            # Look up the user input (Check both possible names just in case)
            user_raw = user_data.get(u_key) or user_data.get(e_key) or ""
            user_val = str(user_raw).lower().strip()
            
            # Look up the AI extraction
            ext_val = str(extracted_data.get(e_key) or "").lower().strip()
            
            is_match = False
            
            # Only run the match logic if the user actually typed something
            if user_val and ext_val:
                if e_key in ["boid", "contact"]:
                    # Phone/BOID: Strip everything except pure numbers
                    u_digits = ''.join(filter(str.isdigit, user_val))
                    e_digits = ''.join(filter(str.isdigit, ext_val))
                    is_match = (u_digits == e_digits) and len(u_digits) > 0
                    
                elif e_key == "citizenship":
                    # Citizenship: Check if user input is INSIDE the extracted text
                    is_match = user_val in ext_val
                    
                else:
                    # Name/DOB: Exact match (ignoring double spaces)
                    is_match = user_val.replace("  ", " ") == ext_val.replace("  ", " ")
                    
            # Save the result using the display key (e_key)
            verification[e_key] = {
                "match": is_match, 
                "extracted": extracted_data.get(e_key) or "Not Found"
            }
            
        return {
            "status": "success",
            "document_type": "demat",
            "extracted_data": extracted_data,
            "verification": verification
        }