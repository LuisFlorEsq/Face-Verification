from pydantic import BaseModel
from typing import Optional, List


class FaceRequestModel(BaseModel):
    img: str  # Base64 string or image URL or file path for the image to be processed
    model_name: Optional[str] = "VGG-Face" # Model name for feature extraction
    detector_backend: Optional[str] = "opencv" # Backend detector
    enforce_detection: Optional[bool] = True # Enforce detection of faces in the image
    align: Optional[bool] = True # Align images before processing (face alignment)
    anti_spoofing: Optional[bool] = False # Not relevant
    max_faces: Optional[int] = None # Not relevant


class VerifyRequestModel(BaseModel):
    img1: str # First image for verification
    img2: str # Second image for verification
    model_name: Optional[str] = "VGG-Face" # Model name for feauture extraction
    detector_backend: Optional[str] = "opencv" # Backend detector
    distance_metric: Optional[str] = "cosine" # Distance metric for comparison
    enforce_detection: Optional[bool] = True # Enforce detection of faces in the image
    align: Optional[bool] = True # Align images before processing (face alignment)
    anti_spoofing: Optional[bool] = False # Not relevant
