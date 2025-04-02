from fastapi import APIRouter, HTTPException
from app.services.face_services import represent, verify
from app.models.face_models import FaceRequestModel, VerifyRequestModel

router = APIRouter()


@router.post("/represent")
async def face_represent(face_request: FaceRequestModel):
    try:
        result = represent(
            img=face_request.img,
            model_name=face_request.model_name,
            detector_backend=face_request.detector_backend,
            enforce_detection=face_request.enforce_detection,
            align=face_request.align,
            anti_spoofing=face_request.anti_spoofing,
            max_faces=face_request.max_faces,
        )
        return result
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))


@router.post("/verify")
async def face_verify(verify_request: VerifyRequestModel):
    try:
        result = verify(
            img1=verify_request.img1,
            img2=verify_request.img2,
            model_name=verify_request.model_name,
            detector_backend=verify_request.detector_backend,
            distance_metric=verify_request.distance_metric,
            enforce_detection=verify_request.enforce_detection,
            align=verify_request.align,
            anti_spoofing=verify_request.anti_spoofing,
        )
        return result
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))