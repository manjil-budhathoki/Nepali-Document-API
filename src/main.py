import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from contextlib import asynccontextmanager

from src.ml_engine import load_models
from src.pipeline import process_document


# Server startup logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # load YOLO and OCR models into memory at startup
    load_models()
    yield
    print("Shutting down...")

app = FastAPI(
    title="Nepali Document Processing API",
    version="1.0.0",
    description="API for processing Nepali documents like Citizenship and Demat using AI models.",
    lifespan=lifespan
)

# API endpoint for document upload and processing

@app.post("/verify")
async def verify_document(
    file: UploadFile = File(...),
    name: str = Form(None),
    citizenship_number: str = Form(None),
    dob: str = Form(None),
    boid: str = Form(None),
    contact_number: str = Form(None)
):
    """
    Upload an image of a document along with  user-provided data.
    The system will classify , OCR, and verify the data.
    """

    # Read the fiel Bytes
    contents  = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    
    # convert Bytes directly to OpenCV Numpy Array
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")
    
    # Package User Imput Data
    user_data = {
        "name": name,
        "citizenship": citizenship_number,
        "dob": dob,
        "boid": boid,
        "contact": contact_number
    }
    
    # Pipeline
    try:
        result = process_document(image, user_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")