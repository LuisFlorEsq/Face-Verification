from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class RepresentRequest(BaseModel):
    model_name: str = "Facenet"
    detector_backend: str = "fastmtcnn"
    enforce_detection: bool = True
    align: bool = True
    anti_spoofing: bool = False
    max_faces: Optional[int] = None

class RepresentResponse(BaseModel):
    results: List[Dict[str, Any]]