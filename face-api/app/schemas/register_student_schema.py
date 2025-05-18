from typing import Optional
from pydantic import BaseModel
from app.config.settings import Config

class RegisterRequest(BaseModel):
    student_id: str
    name: str
    model_name: str = Config.DEFAULT_MODEL_NAME
    detector_backend: str = Config.DEFAULT_DETECTOR_BACKEND
    enforce_detection: bool = Config.ENFORCE_DETECTION
    align: bool = Config.ALIGN_FACES
    
class RegisterResponse(BaseModel):
    message: str
    student_id: str