import cv2
import numpy as np
import requests
from typing import Union, List, Optional, Dict, Any

from fastapi import UploadFile, HTTPException
from starlette.datastructures import UploadFile

from deepface import DeepFace
from app.utils.verification import verify


def load_image(image: Union[UploadFile, str]) -> np.ndarray:
    """
    Load an image from a file path, URL, or UploadFile object.

    Args:
        image (Union[UploadFile, str]): The image to load, either as an UploadFile object or a string (file path or URL).

    Returns:
        _np.ndarray_: The loaded image as a NumPy array (BGR format).
    """
    
    # print(f"[DEBUG] load_image got: {type(image)}")

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
    
    try:
        
        image = load_image(img)
        embedding_objs = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        
        return {"results": embedding_objs}
    
    except Exception as err:
        
        import traceback
        tb_str = traceback.format_exc()
        
        raise HTTPException(status_code=400, detail=f"Exception while representing: {str(err)}\n{tb_str}")


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