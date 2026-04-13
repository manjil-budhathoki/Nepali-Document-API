# streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import io
import base64

# --- CONFIGURATION ---
st.set_page_config(page_title="Doc Nexus | AI Verification", page_icon="🔍", layout="wide")
API_URL = "http://localhost:8000/verify"

# --- SIDEBAR: USER INPUT ---
st.sidebar.title("📝 Enter User Details")
st.sidebar.markdown("Type the details the user provided to verify them against the document.")

user_name = st.sidebar.text_input("Full Name")
user_dob = st.sidebar.text_input("Date of Birth (YYYY-MM-DD)")
user_citizenship = st.sidebar.text_input("Citizenship Number")
user_boid = st.sidebar.text_input("BOID (For Demat)")
user_contact = st.sidebar.text_input("Contact Number (For Demat)")

# --- MAIN PAGE: IMAGE UPLOAD ---
st.title("🔍 Doc Nexus: AI Document Verification")
st.markdown("Upload a Nepali Citizenship or Demat document to instantly extract and verify data.")

uploaded_file = st.file_uploader("Upload Document Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Verification Results")
        
        if st.button("🚀 Run AI Verification", use_container_width=True):
            with st.spinner('AI is analyzing the document... Please wait.'):
                try:
                    # Prepare data for FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {
                        "name": user_name,
                        "dob": user_dob,
                        "citizenship_number": user_citizenship,
                        "boid": user_boid,
                        "contact_number": user_contact
                    }

                    # Call the backend API
                    response = requests.post(API_URL, files=files, data=data, timeout=60)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if result.get("status") == "error":
                            st.error(result.get("message"))
                        else:
                            st.success(f"Document Type Detected: **{result.get('document_type').upper()}**")
                            
                            # --- DISPLAY VERIFICATION MATCHES ---
                            st.markdown("### 📊 Verification Match")
                            verifications = result.get("verification", {})
                            
                            for field, data in verifications.items():
                                # Handle Citizenship format (has 'status' and 'score')
                                if "status" in data:
                                    if data["status"] == "MATCH":
                                        st.success(f"✅ **{field.capitalize()}**: MATCH ({data.get('score')}%)")
                                    elif data["status"] == "PARTIAL":
                                        st.warning(f"⚠️ **{field.capitalize()}**: PARTIAL MATCH ({data.get('score')}%)")
                                    else:
                                        st.error(f"❌ **{field.capitalize()}**: MISMATCH")
                                
                                # Handle Demat format (has 'match' boolean)
                                elif "match" in data:
                                    if data["match"]:
                                        st.success(f"✅ **{field.capitalize()}**: MATCH")
                                    else:
                                        st.error(f"❌ **{field.capitalize()}**: MISMATCH (Found: {data.get('extracted')})")

                            # --- DISPLAY RAW TEXT ---
                            with st.expander("📄 View Raw Extracted Text"):
                                if "extracted_text" in result:
                                    st.write(result["extracted_text"])
                                elif "extracted_data" in result:
                                    st.json(result["extracted_data"])

                            # --- DISPLAY DEBUG IMAGES (YOLO CROP) ---
                            if "debug_images" in result and "yolo_boxes" in result["debug_images"]:
                                with st.expander("🗂️ View AI Cropping (YOLO)"):
                                    img_data = base64.b64decode(result["debug_images"]["yolo_boxes"])
                                    st.image(Image.open(io.BytesIO(img_data)), caption="YOLO Text Block Detection")

                    else:
                        st.error(f"API Error: {response.status_code} - {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("🚨 Could not connect to API. Is your FastAPI server (main.py) running?")
                except requests.exceptions.Timeout:
                    st.error("⏳ Request timed out. The AI models might be taking too long to process.")