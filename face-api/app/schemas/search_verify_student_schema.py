from typing import Union, List
from pydantic import BaseModel
from app.config.settings import Config


class VerifyRequest(BaseModel):
    reference: Union[str, List[float], dict, list]
    test: Union[str, List[float], dict, list]
    model_name: str = Config.DEFAULT_MODEL_NAME
    detector_backend: str = Config.DEFAULT_DETECTOR_BACKEND
    distance_metric: str = Config.DEFAULT_DISTANCE_METRIC
    enforce_detection: bool = Config.ENFORCE_DETECTION
    align: bool = Config.ALIGN_FACES
    

class VerifyResponse(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str
    detector_backend: str
    similarity_metric: str