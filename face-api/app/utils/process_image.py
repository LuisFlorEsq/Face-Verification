import cv2
import numpy as np
import requests
from fastapi import HTTPException
from starlette.datastructures import UploadFile
from typing import Union

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