import cv2
import numpy as np
import requests
from typing import Union, List, Optional, Dict, Any

from fastapi import UploadFile, HTTPException
from starlette.datastructures import UploadFile

from deepface import DeepFace
import tensorflow as tf
from app.utils.verification import verify

# Chroma DB imports

import chromadb
from chromadb.utils import embedding_functions
import os

# Memory and CPU tracing
import logging
import psutil

# Disable GPU usage to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
tf.config.set_visible_devices([], 'GPU')  # Force CPU usage

# Chroma DB initialization

CHROMA_PATH = os.getenv("CHROMA_PERSISTENT_PATH", "./students_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="student_embeddings",
    embedding_function=embedding_function
)

# # Logging configuration
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


def log_resources(prefix: str):
    """
    Log memory and CPU usage.

    Args:
        prefix (str): Prefix for the log message.
    """
    
    process = psutil.Process()
    mem_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=0.1)
    logger.debug(f"{prefix} - Memory Usage: {mem_info.rss / (1024**2):.2f} MB, CPU: {cpu_percent:.2f}%")
    


def load_image(image: Union[UploadFile, str]) -> np.ndarray:
    """
    Load an image from a file path, URL, or UploadFile object.

    Args:
        image (Union[UploadFile, str]): The image to load, either as an UploadFile object or a string (file path or URL).

    Returns:
        _np.ndarray_: The loaded image as a NumPy array (BGR format).
    """
    
    print(f"[DEBUG] load_image got: {type(image)}")

    # UploadFile (direct image upload)
    if isinstance(image, UploadFile):
        
        image.file.seek(0)  # Rewind the stream just in case
        img_bytes = image.file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # print(f"[DEBUG] Read {len(img_bytes)} bytes")
        # print(f"[DEBUG] img is None? {img is None}")

    # Url or file path
    elif isinstance(image, str):
        
        # print(f"[DEBUG] load image got a string: {image}")
        
        if image.startswith("http"):
            
            response = requests.get(image)
            img_array = np.frombuffer(response.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
        else:
            
            img = cv2.imread(image)
            
    else:
        
        raise HTTPException(status_code=400, detail="Unsupported image type. Must be UploadFile or a string (URL or file path).")

    # Ensure correct image load, decoding and conversion
    
    if img is not None:
        
        if len(img.shape) == 3 and img.shape[2] == 4: # PNG format with alpha
            
            # print(f"[DEBUG] Image has 4 channels")
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
        elif len(img.shape) == 3:  # Regular color image
            
            # print(f"[DEBUG] Image has 3 channels")
            pass
            
        else:
            
            raise HTTPException(status_code=400, detail="Unsupported image format or grayscale image.")
        
    else:
        
        raise HTTPException(status_code=400, detail="Failed to load image!")
    
    return img


def represent_student(
    img: Union[UploadFile, str],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None
) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
    """
    Get the face embeddings for the given image.

    Args:
        img (Union[UploadFile, str]): Image can be a file path, URL or UploadFile object.
        model_name (str, optional): The face recognition model to generate embeddings. Defaults to "VGG-Face".
        detector_backend (str, optional): Face detection model. Defaults to "opencv".
        enforce_detection (bool, optional): Perform face detection before obtain embedding. Defaults to True.
        align (bool, optional): Align the detected face. Defaults to True.

    Returns:
        results (List[Dict[str, Any]] or List[Dict[str, Any]]): A list of dictionaries.
        Result type becomes List of List of Dict if batch input passed.
    """
    
    # # log_resources("Before represent_student")
    
    try:
        
        image = load_image(img)
        embedding_objs = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=1,
        )
        
        # # log_resources("After represent_student")
        
        return {"results": embedding_objs}
    
    except Exception as err:
        
        import traceback
        tb_str = traceback.format_exc()
        
        raise HTTPException(status_code=400, detail=f"Exception while representing: {str(err)}\n{tb_str}")


def register_student(
    student_id: str,
    name: str,
    img: Union[UploadFile, str],
    model_name: str = "Facenet",
    detector_backend: str = "ssd",
    enforce_detection: bool = True,
    align: bool = True
) -> Dict[str, Any]:
    """
    Register a student by storing their face embedding in ChromaDB.

    Args:
        student_id (str): The current student ID to be enrolled.
        name (str): The name of the student.
        img (Union[UploadFile, str]): The reference image of the student, can be a file path, URL or UploadFile object.
        model_name (str, optional): Embedding generation model. Defaults to "Facenet".
        detector_backend (str, optional): Face detection model. Defaults to "ssd".
        enforce_detection (bool, optional): Ensure to detect a face before embedding generation. Defaults to True.
        algin (bool, optional): Align the detected face before embedding generation. Defaults to True.

    Returns:
        Dict[str, Any]: Registration status and details.
    """
    
    # # log_resources("Before register_student")
    
    try:
        
        # Check if student_id already exists in the database
        
        existing_student = collection.get(ids=[student_id])
        
        if existing_student["ids"]:
            raise HTTPException(status_code=400, detail="Student ID already exists.")
        
        
        # Generate embedding
        image = load_image(img)
        
        embedding_obj = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align
        )
        
        # print(f"[DEBUG]: Embedding object", embedding_obj)
        # print(f"[DEBUG]: Embedding", embedding_obj[0]["embedding"])
        
        embedding = embedding_obj[0]["embedding"]
        
        # Store the embedding in ChromaDB
        
        collection.add(
            embeddings=[embedding],
            metadatas=[{"student_id": student_id, "name": name}],
            ids=[student_id]
            
        )
        
        # # log_resources("After register_student")
        
        return {
            "message": f"Student {name} registered successfully.", 
            "student_id": student_id
        }
        
    
    except Exception as err:
        
        import traceback
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Exception while registeing: {str(err)}\n{tb_str}")
    

def search_verify_student(
    student_id: str,
    reference_img: Union[UploadFile, str],
    model_name: str = "Facenet",
    detector_backend: str = "ssd",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
) -> Dict[str, Any]:
    """
    Search for a student's embedding in the database and verify it against a reference image.

    Args:
        student_id (str): The student ID to search for.
        reference_img (Union[UploadFile, str]): The reference image to verify against.
        model_name (str, optional): Embedding generation model. Defaults to "Facenet".
        detector_backend (str, optional): Face detection model. Defaults to "ssd".
        enfoce_detection (bool, optional): Ensure to detect a face before embedding generation. Defaults to True.
        align (bool, optional): Align the detected face before embedding generation. Defaults to True.

    Returns:
        Dict[str, Any]: Verification result including distance and status.
    """
    
    # # log_resources("Before search_verify_student")
    
    try:
        
        reference_img = load_image(reference_img)
        
        # Query ChromaDB for the stored embedding of the student
        results = collection.get(
            ids=[student_id],
            include=["metadatas","embeddings"]
        )
        
        # print(f"[DEBUG] Search results: {results}")
        
        if not results["ids"]:
            
            raise HTTPException(status_code=404, detail="Student ID not found.")
        
        reference_embedding = results["embeddings"][0].tolist()
        
        # print(f"[DEBUG] Reference embedding: {reference_embedding}")
        # print(f"[DEBUG] Format type: {type(reference_embedding)}")
        
        
        # # log_resources("After search_verify_student")
        
        # Verify both embeddings
        
        return verify(
            img1_path=reference_embedding,
            img2_path=reference_img,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align  
        )
        
    
    except Exception as err:
        
        import traceback
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Exception while verifying: {str(err)}\n{tb_str}")


def verify_student(
    reference: Union[UploadFile, str, List[float], dict, list],
    test: Union[UploadFile, str, List[float], dict, list],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True
):
    """
    Verify if two images or embeddings are of the same person (student) based on the selected face recognition model and distance metric.

    Args:
        reference (Union[UploadFile, str, List[float], dict, list]): Image or precomputed embedding of the reference person.
        test (Union[UploadFile, str, List[float], dict, list]): Image or precomputed embedding of the test person.
        model_name (str, optional): Face recognition model. Defaults to "VGG-Face".
        detector_backend (str, optional): Face detection model. Defaults to "opencv".
        distance_metric (str, optional): Distance metric used to calculate the similarity. Defaults to "cosine".
        enforce_detection (bool, optional): Perform face detection before compute the embeddings. Defaults to True.
        align (bool, optional): Align the detected faces. Defaults to True

    Returns:
        _type_: _description_
    """
    
    try:
        def preprocess_input(input_data):
            if isinstance(input_data, (UploadFile, str)):
                return load_image(input_data)
            elif isinstance(input_data, list) and all(isinstance(x, (float, int)) for x in input_data):
                return input_data
            elif isinstance(input_data, dict) and "results" in input_data:
                # Handle output from represent_student
                if input_data["results"] and isinstance(input_data["results"][0], dict) and "embedding" in input_data["results"][0]:
                    return input_data["results"][0]["embedding"]
                raise ValueError("Invalid embedding object format")
            elif isinstance(input_data, list) and input_data and isinstance(input_data[0], dict) and "embedding" in input_data[0]:
                # Handle list of embedding objects
                return input_data[0]["embedding"]
            else:
                raise ValueError("Invalid input format. Must be an image, embedding list, or embedding object")

        # Preprocess inputs
        img1_path = preprocess_input(reference)
        img2_path = preprocess_input(test)

        # Adjust detector_backend and enforce_detection for embeddings
        if isinstance(img1_path, list) and isinstance(img2_path, list):
            detector_backend = "skip"
            enforce_detection = False

        verification = verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align
        )
        
        return verification
    
    except Exception as err:
        import traceback
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Exception while verifying: {str(err)}\n{tb_str}")