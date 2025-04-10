import cv2
import numpy as np
import requests
from fastapi import UploadFile, HTTPException
from starlette.datastructures import UploadFile
from typing import Union
from deepface import DeepFace

def load_image(image: Union[UploadFile, str]):
    
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
        
        print(f"[DEBUG] load image got a string: {image}")
        
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

def represent(img, model_name="VGG-Face", detector_backend="opencv", enforce_detection=True, align=True, anti_spoofing=False, max_faces=None):
    
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

def verify(img1, img2, model_name="VGG-Face", detector_backend="opencv", distance_metric="cosine", enforce_detection=True, align=True, anti_spoofing=False):
    
    try:
        
        image1 = load_image(img1)
        image2 = load_image(img2)
        verification = DeepFace.verify(
            img1_path=image1,
            img2_path=image2,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        
        return verification
    
    except Exception as err:
        
        import traceback
        tb_str = traceback.format_exc()
        
        raise HTTPException(status_code=400, detail=f"Exception while verifying: {str(err)}\n{tb_str}")