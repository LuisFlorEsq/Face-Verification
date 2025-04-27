from typing import Optional
from pydantic import BaseModel

class RegisterRequest(BaseModel):
    student_id: str
    name: str
    img_path: Optional[str] = None
    model_name: str = "Facenet"
    detector_backend: str = "fastmtcnn"
    enforce_detection: bool = True
    align: bool = True
    
class RegisterResponse(BaseModel):
    message: str
    student_id: str