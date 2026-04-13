"""
This file handles all the text or string manupulations which includes:
1. Fuzzy matching-
2. Regex Extractions-
3. Nepali Translations-
"""
import re
import unicodedata
import datetime
import nepali_datetime
from difflib import SequenceMatcher
from config import FUZZY_MATCH_THRESHOLD_HIGH, FUZZY_MATCH_THRESHOLD_LOW

# Constants and Maps
NEP_CONSONANT_MAP = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'n', 'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'n',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n', 'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'f', 'ब': 'b', 'भ': 'bh', 'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'b', 'श': 's', 
    'ष': 'sh', 'स': 's', 'ह': 'h', 'क्ष': 'ksh', 'त्र': 'tr', 'ज्ञ': 'gy'
}

NEP_TO_ENG_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Text Normalization
def normalize_unicode(text: str) -> str:
    if not text: return ""
    text = unicodedata.normalize('NFKC', text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != 'C')

def normalize_to_eng_digits(text:str) -> str:
    return text.translate(NEP_TO_ENG_DIGITS) if text else ""

def clean_and_repair_text(raw_text: str) -> str:
    """Cleans up OCR text spacing and formatting."""
    if not raw_text: return ""
    text = normalize_unicode(raw_text)
    # Fix spacing issues like "Year2000" -> "Year 2000"
    text = re.sub(r'([a-zA-Z\u0900-\u097F])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z\u0900-\u097F])', r'\1 \2', text)
    keywords = ["Year", "Month", "Day", "नाम", "थर", "जन्म", "मिति", "नं"]
    for kw in keywords:
        text = re.sub(f"({kw})", r" \1 ", text, flags=re.IGNORECASE)
    text = re.sub(r'[:;|।!]', ' : ', text)
    return re.sub(r'\s+', ' ', text).strip()



def extract_demat_fields(text: str) -> dict:
    """Extracts fields from raw Demat OCR text using Regex."""
    return {
        "name": (re.search(r"Name\s+(.+)", text, re.I) or [None, None])[1],
        "boid": (re.search(r"BOID\s+(\d+)", text, re.I) or [None, None])[1],
        "dob": (re.search(r"Date Of Birth\s+(\d{4}-\d{2}-\d{2})", text, re.I) or [None, None])[1],
        "citizenship": (re.search(r"Citizenship Number\s+(.+)", text, re.I) or [None, None])[1],
        "contact": (re.search(r"Contact Number\s+(\d+)", text, re.I) or [None, None])[1]
    }

def get_consonant_skeleton(text: str, script: str = "english") -> str:
    skeleton = ""
    text = text.lower().strip()
    if script == "nepali":
        for char in text:
            if char in NEP_CONSONANT_MAP:
                skeleton += NEP_CONSONANT_MAP[char]
    else:
        skeleton = "".join(char for char in re.sub(r'[^a-z]', '', text) if char not in "aeiou")
    return skeleton

def verify_name(user_name: str, ocr_text: str) -> dict:
    if not user_name: return {"score": 0, "status": "MISMATCH"}
    
    if user_name.lower() in ocr_text.lower():
        return {"score": 100, "status": "MATCH", "method": "Exact English"}

    u_skeleton = get_consonant_skeleton(user_name, "english")
    nepali_chars = "".join(re.findall(r'[\u0900-\u097F]+', ocr_text))
    ocr_skeleton = get_consonant_skeleton(nepali_chars, "nepali")
    
    if u_skeleton and ocr_skeleton:
        if u_skeleton in ocr_skeleton:
            return {"score": 100, "status": "MATCH", "method": "Consonant Skeleton"}
        
        score = int(SequenceMatcher(None, u_skeleton, ocr_skeleton).ratio() * 100)
        if score > FUZZY_MATCH_THRESHOLD_HIGH:
            return {"score": score, "status": "MATCH", "method": "Fuzzy Skeleton"}
        if score > FUZZY_MATCH_THRESHOLD_LOW:
            return {"score": score, "status": "PARTIAL", "method": "Fuzzy Skeleton"}

    return {"score": 0, "status": "MISMATCH", "method": "None"}

def verify_id_number(user_id: str, ocr_text: str) -> dict:
    if not user_id: return {"score": 0, "status": "MISMATCH"}
    
    ocr_digits = re.sub(r'\D', '', normalize_to_eng_digits(ocr_text))
    u_digits = re.sub(r'\D', '', str(user_id))
    
    if u_digits and u_digits in ocr_digits:
        return {"score": 100, "status": "MATCH"}
    return {"score": 0, "status": "MISMATCH"}

def verify_dob(user_dob: str, ocr_text: str) -> dict:
    if not user_dob: return {"score": 0, "status": "MISMATCH"}
    
    full_corpus = normalize_to_eng_digits(ocr_text)
    try:
        y_ad, m_ad, d_ad = map(int, user_dob.split('-'))
        ad_date = datetime.date(y_ad, m_ad, d_ad)
        bs_date = nepali_datetime.date.from_datetime_date(ad_date)
        y_bs, m_bs, d_bs = bs_date.year, bs_date.month, bs_date.day
        
        ad_tokens = {str(y_ad), f"{m_ad:02}", f"{d_ad:02}"}
        bs_tokens = {str(y_bs), f"{m_bs:02}", f"{d_bs:02}", str(m_bs), str(d_bs)}
    except Exception:
        return {"score": 0, "status": "ERROR", "message": "Invalid DOB format (Expected YYYY-MM-DD)"}

    if len([t for t in bs_tokens if t in full_corpus]) >= 2 and str(y_bs) in full_corpus:
        return {"score": 100, "status": "MATCH", "matched_type": "BS"}
        
    if len([t for t in ad_tokens if t in full_corpus]) >= 2 and str(y_ad) in full_corpus:
        return {"score": 100, "status": "MATCH", "matched_type": "AD"}
        
    return {"score": 0, "status": "MISMATCH"}