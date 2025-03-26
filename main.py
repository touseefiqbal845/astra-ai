import mediapipe as mp
import cv2
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, Request, Depends
import re
import google.cloud.vision as vision
import base64
from fastapi import Depends
import numpy as np
from sqlalchemy.orm import Session
from test_database import SessionLocal, FaceEmbedding
from fastapi.responses import JSONResponse
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter
from qdrant_client import QdrantClient
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from OCR_utils import detect_text, extract_cnic_details, extract_face


app = FastAPI()

QDRANT_HOST =  "0ef2bd19-7221-4f4e-b138-a8f7dfbdeeeb.us-east4-0.gcp.cloud.qdrant.io" 
QDRANT_PORT = 6333
COLLECTION_NAME = "face_embeddings"
embedding_dim = 2048
client = QdrantClient(url = f"https://{QDRANT_HOST}:{QDRANT_PORT}",
                      api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.nluANc6tk-JUaAui496x5SZ5wo0fLrlIl7oeX9Nf0wc", ) #host=QDRANT_HOST, port=QDRANT_PORT

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def decode_base64_image(base64_string):
    """
    Convert a base64-encoded string to an OpenCV image.

    Args:
        base64_string (str): The base64 string representation of the image.

    Returns:
        numpy.ndarray: The decoded image in OpenCV format.

    Raises:
        HTTPException: If the input string is not a valid base64 image format.
    """
    try:
        # Decode the base64 string to binary image data
        image_data = base64.b64decode(base64_string)
        
        # Convert binary data to a NumPy array
        np_arr = np.frombuffer(image_data, np.uint8)
        
        # Decode the NumPy array into an OpenCV image
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        # Raise HTTPException if decoding fails
        raise HTTPException(status_code=400, detail="Invalid base64 image format")

def initialize_qdrant():
    """
    Initializes the Qdrant collection for storing face embeddings.

    Creates a new collection in Qdrant with the specified name and vector configuration.
    """
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=embedding_dim,  # The size of the vectors in the collection
                distance=Distance.COSINE,  # The distance metric used for similarity search
            ),
        )
        print(f"Qdrant collection {COLLECTION_NAME} created.")

initialize_qdrant()


# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

# Load a pre-trained model
model = ResNet50(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))


def generate_embedding(image):
    """
    Generates an embedding for a given image using the pre-trained model.

    Args:
        image: A 3D numpy array representing the image.

    Returns:
        A 1D numpy array representing the embedding of the image.
    """

    # Resize the image to the input size of the model (224x224)
    resized_image = cv2.resize(image, (224, 224))
    
    # Convert the image to an array and add a batch dimension (required by Keras)
    resized_image = img_to_array(resized_image)
    resized_image = np.expand_dims(resized_image, axis=0)
    
    # Preprocess the image using the Keras `preprocess_input` function
    resized_image = preprocess_input(resized_image)
    
    # Generate the embedding of the image using the pre-trained model
    
    embedding = model.predict(resized_image).flatten()
    
    # Normalize the embedding vector to have unit length
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def is_face_already_exist(new_embedding, threshold=0.5):
    """
    Checks if a face already exists in the Qdrant collection based on the cosine similarity of embeddings.

    Args:
        new_embedding (numpy.ndarray): The embedding vector to be checked.
        threshold (float): The similarity threshold above which the face is considered to exist.

    Returns:
        bool: True if the face already exists, False otherwise.
        float: The cosine similarity of the closest match if the face exists, 0 otherwise.
        int: The ID of the closest match if the face exists, -1 otherwise.
    """
    print("Checking if face exists in Qdrant collection...")
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=new_embedding.tolist(),
        limit=1,
    )
    print("Search result:", search_result)

    if search_result and search_result[0].score > threshold:
        print("Face already exists in Qdrant collection.")
        return True, search_result[0].score, search_result[0].id
    print("Face does not exist in Qdrant collection.")
    return False, 0, -1

def append_embedding(new_embedding):
    """
    Appends a single embedding to the Qdrant collection.

    Args:
        new_embedding (numpy.ndarray): The embedding vector to be added.
    """
    point_id = np.random.randint(1, 1e6)  # Generate a unique ID for the embedding
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(id=point_id, vector=new_embedding.tolist()),  # Convert embedding to list and add to Qdrant
        ],
    )
    print(f"Embedding added with ID {point_id}")  # Log the addition of the embedding

def is_frame_noisy(frame, threshold=5):
    """
    Checks if the frame contains significant noise using Laplacian variance.
    
    Args:
        frame (numpy.ndarray): Input image frame.
        threshold (float): The variance threshold below which the frame is considered noisy.
    
    Returns:
        bool: True if the frame contains noise, False otherwise.
        float: The calculated Laplacian variance.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian operator to detect edges and high-frequency noise
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
    
    # Calculate the variance of the Laplacian
    laplacian_variance = laplacian.var()
    
    # Check if the variance is below the threshold
    if laplacian_variance < threshold:
        return True, laplacian_variance  # Noisy frame
    return False, laplacian_variance  # Clean frame

def is_frame_blurry(frame, threshold=50):
    """
    Checks if the frame is blurry using Laplacian variance.
    
    Args:
        frame (numpy.ndarray): Input image frame.
        threshold (float): The variance threshold below which the frame is considered blurry.
    
    Returns:
        bool: True if the frame is blurry, False otherwise.
        float: The calculated Laplacian variance.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian operator to detect edges
    laplacian = cv2.Laplacian(gray_frame, cv2.CV_64F)
    
    # Calculate the variance of the Laplacian
    laplacian_variance = laplacian.var()
    
    # Check if the variance is below the threshold
    if laplacian_variance < threshold:
        return True, laplacian_variance  # Blurry frame
    return False, laplacian_variance  # Sharp frame



@app.post("/register_face/")
async def register_face(request: Request, db: Session = Depends(get_db)):
    try:
        data = await request.json()
        base64_image = data.get("image_base64")
        user_id = 2  # Retrieve user_id from request
        if not base64_image or user_id is None:
            return JSONResponse(content={"status": "error", "message": "Missing 'image_base64' or 'user_id' field."}, status_code=400)

        image = decode_base64_image(base64_image)
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return {"status": "error", "message": "No face detected. Please try again."}

        embedding = generate_embedding(image)
        exists, score, face_id = is_face_already_exist(embedding)
        if exists:
            return {"status": "success", "message": "Face already registered", "user_id": user_id, "point_id": str(face_id)}

        new_face = FaceEmbedding(user_id=user_id, embedding=embedding)
        db.add(new_face)
        db.commit()

        point_id = append_embedding(embedding)
        return {"status": "success", "message": "Face registered successfully", "user_id": user_id, "point_id": point_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/extract_info/")
async def extract_info(request: Request):
    try:
        data = await request.json()
        extracted_image = data.get("extracted_image_base64")
        
        user_id = data.get("user_id")
        if not extracted_image:
            return JSONResponse(content={"status": "error", "message": "Missing 'image_base64' field."}, status_code=400)

        image = decode_base64_image(extracted_image)
        data = detect_text(image)
        user_info = extract_cnic_details(data)

        face_extraction = extract_face(image)
        face_embedds = generate_embedding(face_extraction)
        exists, score, face_id = is_face_already_exist(face_embedds)
        if exists:
            return JSONResponse(content={"status": "success", "message": "Face Match", "user_id": user_id, "point_id": str(face_id), "data": user_info})
        else:
            return JSONResponse(content={"status": "fail", "message": "Face Not Match", "user_id": user_id, "data": user_info})
        
        # return JSONResponse(content={"status": "success", "user_id": user_id, "data": user_info})
        
        

    except Exception as e:
        return {"status": "error", "message": str(e)}


# @app.post("/match_faces/")
# async def match_faces(request: Request):
#     try:
        
#     except Exception as e:
#         return {"status": "error", "message": str(e)}