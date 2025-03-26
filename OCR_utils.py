import re
from google.cloud import vision
import os
import numpy as np
from dotenv import load_dotenv
import cv2
# Load environment variables
load_dotenv()

# Set Google Cloud credentials from .env file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv("GOOGLE_CLOUD_CREDENTIALS_PATH")


client = vision.ImageAnnotatorClient()



def detect_text(image):
    """Detects text in an image using Google Vision API OCR."""
    # Convert NumPy array to bytes (Google Vision API expects bytes)
    _, encoded_image = cv2.imencode('.jpg', image)
    image_bytes = encoded_image.tobytes()

    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        return texts[0].description  # Return extracted text
    else:
        return None  # No text detected
    
def extract_face(image):
    """Detects and extracts the face from an image and returns it as a NumPy array."""
    _, encoded_image = cv2.imencode('.jpg', image)
    content = encoded_image.tobytes()
    

    image = vision.Image(content=content)
    face_response = client.face_detection(image=image)

    if not face_response.face_annotations:
        return None

    # Read the image with OpenCV
    image_cv = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

    # Get face bounding box
    face = face_response.face_annotations[0]
    vertices = face.bounding_poly.vertices
    x1, y1 = vertices[0].x, vertices[0].y
    x2, y2 = vertices[2].x, vertices[2].y

    # Crop the face
    face_image = image_cv[y1:y2, x1:x2]

    return face_image

# def extract_cnic_details(ocr_text):
    """Extracts key details from CNIC text dynamically with improved accuracy."""
    if not ocr_text:
        return {"Error": "No text detected in the image"}

    extracted_info = {}

    # Normalize text (remove extra spaces and newlines)
    lines = [line.strip() for line in ocr_text.split("\n") if line.strip()]

    # Extract Name
    extracted_info["Name"] = "Not Found"
    for i, line in enumerate(lines):
        if "National Identity Card" in line and i + 1 < len(lines):
            extracted_info["Name"] = lines[i + 1]  # Name is below this
            break
        elif "Father Name" in line and i > 0:
            extracted_info["Name"] = lines[i - 1]  # Name is above this
            break

    # Extract Father's Name
    extracted_info["Father Name"] = "Not Found"
    for i, line in enumerate(lines):
        if "Father Name" in line and i + 1 < len(lines):
            extracted_info["Father Name"] = lines[i + 1]  # Next line
            break

    # Extract CNIC Number
    cnic_match = re.search(r"\b\d{5}-\d{7}-\d{1}\b", ocr_text)
    extracted_info["CNIC Number"] = cnic_match.group(0) if cnic_match else "Not Found"

    # Extract Dates with Improved Logic
    extracted_info["Date of Birth"] = "Not Found"
    extracted_info["Date of Issue"] = "Not Found"
    extracted_info["Date of Expiry"] = "Not Found"

    for i, line in enumerate(lines):
        if "Date of Birth" in line and i + 1 < len(lines):
            extracted_info["Date of Birth"] = re.search(r"\d{2}[-.]\d{2}[-.]\d{4}", lines[i + 1]).group(0)

        if "Date of Issue" in line and i + 1 < len(lines):
            extracted_info["Date of Issue"] = re.search(r"\d{2}[-.]\d{2}[-.]\d{4}", lines[i + 1]).group(0)

        if "Date of Expiry" in line and i + 1 < len(lines):
            extracted_info["Date of Expiry"] = re.search(r"\d{2}[-.]\d{2}[-.]\d{4}", lines[i + 1]).group(0)

    # Extract Gender
    extracted_info["Gender"] = "Not Found"
    for i, line in enumerate(lines):
        if "Gender" in line and i + 1 < len(lines):
            gender_text = lines[i + 1].strip()
            extracted_info["Gender"] = "Male" if "M" in gender_text else "Female" if "F" in gender_text else "Unknown"

    # Extract Nationality
    extracted_info["Nationality"] = "Pakistan" if "Pakistan" in ocr_text else "Not Found"
    extracted_info["Country of Residence"] = extracted_info["Nationality"]

    return extracted_info
