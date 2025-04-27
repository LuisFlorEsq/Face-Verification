from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.schemas.represent_student_schema import RepresentRequest, RepresentResponse
from app.services.post_represent_service import represent_student

router = APIRouter(tags=["Represent Student"])

@router.post("/represent_student", response_model=RepresentResponse)
async def face_represent_student(
    img: Optional[UploadFile] = File(None),
    img_path: Optional[str] = Form(None),
    model_name: str = Form("Facenet"),
    detector_backend: str = Form("fastmtcnn"),
    enforce_detection: bool = Form(True),
    align: bool = Form(True),
    anti_spoofing: bool = Form(False),
    max_faces: Optional[int] = Form(None)
):
    if img is None and img_path is None:
        raise HTTPException(status_code=400, detail="Either 'img' or 'img_path' must be provided.")
    
    img_input = img if img is not None else img_path
    
    return await represent_student(
        img=img_input,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces,
    )