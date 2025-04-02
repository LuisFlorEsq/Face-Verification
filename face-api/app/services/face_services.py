import traceback
from deepface import DeepFace
from typing import Union, Optional
import numpy as np
from app.models.face_models import FaceRequestModel, VerifyRequestModel


def represent(
    img: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None,
):
    try:
        # Using DeepFace's represent function
        embedding_objs = DeepFace.represent(
            img_path=img,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        return {"results": embedding_objs}
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1: Union[str, np.ndarray],
    img2: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    anti_spoofing: bool = False,
):
    try:
        # Using DeepFace's verify function
        verification = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return verification
    except Exception as err:
        tb_str = traceback.format_exc()
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400