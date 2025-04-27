from typing import Union, List
from pydantic import BaseModel


class VerifyRequest(BaseModel):
    reference: Union[str, List[float], dict, list]
    test: Union[str, List[float], dict, list]
    model_name: str = "Facenet"
    detector_backend: str = "fastmtcnn"
    distance_metric: str = "cosine"
    enforce_detection: bool = True
    align: bool = True
    

class VerifyResponse(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str
    detector_backend: str
    similarity_metric: str