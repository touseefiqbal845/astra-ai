from __future__ import annotations
import cv2
import pickle
import os
import mediapipe as mp
import numpy as np
import uuid
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import faiss
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, Filter

### USE THE FAISS INDEX MODULE FOR LOCAL TESTING ###

# File to store all embeddings
embedding_file = "faiss_index.index"
QDRANT_HOST =  "0ef2bd19-7221-4f4e-b138-a8f7dfbdeeeb.us-east4-0.gcp.cloud.qdrant.io" 
QDRANT_PORT = 6333
COLLECTION_NAME = "face_embeddings"

# # Initialize FAISS index
embedding_dim = 2048  
# index = None

# def initialize_faiss_index():
#     """
#     Initializes the global FAISS index variable. If an existing FAISS index file is found, it loads the index from the file.
#     Otherwise, it creates a new FAISS index using L2 distance.

#     This function assumes that the global variables `index`, `embedding_file`, and `embedding_dim` are defined.
#     """
#     global index
#     if os.path.exists(embedding_file):
#         index = faiss.read_index(embedding_file)  # Load existing FAISS index
#     else:
#         index = faiss.IndexFlatL2(embedding_dim)  # Initialize FAISS index (L2 distance)
#         print("Initialized new FAISS index.")

# Initialize Qdrant client
client = QdrantClient(url = f"https://{QDRANT_HOST}:{QDRANT_PORT}",
                      api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.nluANc6tk-JUaAui496x5SZ5wo0fLrlIl7oeX9Nf0wc" ) #host=QDRANT_HOST, port=QDRANT_PORT

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

# Check if a face exists in Qdrant
def is_face_already_exist(new_embedding, threshold=0.2):
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

# def save_faiss_index():

#     """Saves the current FAISS index to the file specified by `embedding_file`. 
#     This function assumes that the global variable `index` is defined and is a valid FAISS index.
#     """
#     faiss.write_index(index, embedding_file)

# def append_embedding(new_embedding):
#     """Appends a single embedding to the FAISS index."""
#     global index
#     new_embedding = np.array([new_embedding], dtype=np.float32)  # Ensure the embedding is the correct type
#     index.add(new_embedding)  # Add the new embedding to the FAISS index
#     save_faiss_index()  # Save the updated index
# initialize_faiss_index()
# def is_face_already_exist(new_embedding, threshold=500.0):
#     """
#     Checks if a face already exists in the collection of embeddings.
#     """
#     if index is None:
#         print("Error: FAISS index is not initialized.")
#         return False, 0, -1

#     new_embedding = np.array([new_embedding], dtype=np.float32)
#     distances, indices = index.search(new_embedding, k=1)  # Search for the closest match (k=1)
#     print("\n\n\n")
#     print("==============================")
#     print(distances, indices)
#     print(distances[0][0])
#     print("==============================")
#     if distances[0][0] < threshold:  # Check if the closest distance is below the threshold
#         return True, distances[0][0], indices[0][0]
#     return False, 0, -1

def embedding_distance_check_database(new_embedding):
    """
    Checks if a face already exists in the collection of embeddings based on the cosine similarity of embeddings.

    Args:
        new_embedding (numpy.ndarray): New embedding to be checked.

    Returns:
        bool: True if the face already exists, False otherwise.
    """

    face_exists, similarity, match_index = is_face_already_exist(new_embedding)
    print(face_exists, similarity, match_index)
    if face_exists:
        return True
    append_embedding(new_embedding)
    return False

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


    resized_image = cv2.resize(image, (224, 224))
    resized_image = img_to_array(resized_image)
    resized_image = np.expand_dims(resized_image, axis=0)
    resized_image = preprocess_input(resized_image)
    embedding = model.predict(resized_image)
    return embedding.flatten()

def is_frame_noisy(frame, threshold=10):
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

frame_collection = []


def is_frame_blurry(frame, threshold=100):
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


def show_frames_with_matplotlib(frames):
    """
    Displays the captured frames using Matplotlib in a grid.
    
    Args:
        frames (list): List of frames to display.
    """
    num_frames = len(frames)
    fig, axes = plt.subplots(1, num_frames, figsize=(15, 5))
    for i, ax in enumerate(axes):
        ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for Matplotlib
        ax.set_title(f"Frame {i + 1}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def calculate_frame_differences(frames):
    """
    Calculates the pairwise Euclidean distance between frames.
    
    Args:
        frames (list): List of frames (numpy arrays) to compare.
    
    Returns:
        dict: A dictionary with the pairwise distances between frames.
    """
    num_frames = len(frames)
    distances = {}

    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            # Flatten frames into 1D vectors for comparison
            frame1 = frames[i].flatten()
            frame2 = frames[j].flatten()
            
            # Calculate Euclidean distance
            dist = euclidean(frame1, frame2)
            distances[f"Frame {i + 1} vs Frame {j + 1}"] = dist
    distances = []
    for i in range(num_frames):
        for j in range(i + 1, num_frames):
            diff = frames[i].astype(np.float32) - frames[j].astype(np.float32)
            euclidean_distance = np.linalg.norm(diff)
            distances.append(euclidean_distance)
    embedded_value=generate_embedding(frames[1])
    print(embedded_value)
    flag=embedding_distance_check_database(embedded_value)
    if flag == True:
        print("Face already exists!")
    else:
        print("Face added")
        append_embedding(embedded_value)

    return np.mean(distances) ,  flag

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=1.0)

def create_passport_photo(frame):
    """
    Creates a passport-style photo from a camera frame.
    
    Args:
        frame (numpy.ndarray): Input frame from the camera.
    
    Returns:
        numpy.ndarray: Processed passport-style photo.
    """
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    
    # Detect faces in the frame
    results = face_detection.process(rgb_frame)
    
    if not results.detections:
        print("No face detected.")
        return None
    # Get the bounding box of the first detected face
    annotated_frame = frame.copy()
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        x_min = int(bboxC.xmin * w)
        y_min = int(bboxC.ymin * h)
        box_width = int(bboxC.width * w)
        box_height = int(bboxC.height * h)
        x_max = x_min + box_width
        y_max = y_min + box_height
        try:

            mp_drawing.draw_detection(annotated_frame, detection)
        except:
            pass

        # Expand the bounding box slightly to include shoulders
        x_min = max(0, x_min - int(0.1 * box_width))
        y_min = max(0, y_min - int(0.2 * box_height))
        x_max = min(w, x_max + int(0.1 * box_width))
        y_max = min(h, y_max + int(0.2 * box_height))
        
        # Crop the face region
        cropped_face = frame[y_min:y_max, x_min:x_max]
        
        # Resize to passport size (e.g., 600x600 pixels)
        passport_photo = cv2.resize(cropped_face, (600, 600))
        
        # Set a white background (if needed)
        white_background = np.ones((600, 600, 3), dtype=np.uint8) * 255
        white_background[:passport_photo.shape[0], :passport_photo.shape[1]] = passport_photo
        
        return white_background , annotated_frame

    return None

def save_frames_and_embeddings(frames, embeddings, folder_name):
    """
    Saves frames into a specified folder and returns the frames for visualization.
    """
    os.makedirs(folder_name, exist_ok=True)
    
    saved_frames = []
    for i, frame in enumerate(frames):
        file_path = os.path.join(folder_name, f"{i + 1}.jpg")
        cv2.imwrite(file_path, frame)
        saved_frames.append(frame)
    
    return saved_frames


cap = cv2.VideoCapture(0)  # Open webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Generate the passport photo
    result = create_passport_photo(frame)
    if result is not None:
        white_background, annotated_frame = result
        
    else:
        white_background, annotated_frame = None, None

    # Display the passport photo
    if white_background is not None:
        is_blurry, blur_level = is_frame_blurry(white_background, threshold=12)
        is_noisy, noise_level = is_frame_noisy(white_background, threshold=12)
        if not is_blurry and not is_noisy:
            frame_collection.append(white_background)
            if len(frame_collection) == 3:
                folder_name = str(uuid.uuid4())
                
                # Placeholder embeddings
                embeddings = [np.random.rand(128) for _ in frame_collection]  
                
                saved_frames=save_frames_and_embeddings(frame_collection, embeddings, folder_name)
                print(f"Frames and embeddings saved in folder: {folder_name}")
                x=calculate_frame_differences(saved_frames)
                cv2.putText(frame, str(x), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("distance", frame)
                #show_frames_with_matplotlib(saved_frames)
                
                print(x)
                
                frame_collection = []  # Reset the collection after saving
        text = f"Blur Level: {blur_level:.2f} - {'Blurry' if is_blurry else 'Sharp'}"
        cv2.putText(white_background, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not is_blurry else (0, 0, 255), 2)
        text = f"Noise Level: {noise_level:.2f} - {'Noisy' if is_noisy else 'Clean'}"
        cv2.putText(white_background, text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if not is_noisy else (0, 0, 255), 2)
        cv2.imshow("Face Frame", white_background)
        cv2.imshow("Landmarks Frame",annotated_frame)
        cv2.imshow("Camera Frame", frame)
    else:
        cv2.imshow("Camera Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
