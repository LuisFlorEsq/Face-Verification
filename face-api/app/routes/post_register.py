from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from app.schemas.register_student_schema import RegisterResponse
from app.services.post_register_service import register_student

router = APIRouter(tags=["Register Student"])

@router.post("/register_student", response_model=RegisterResponse)
async def face_register_student(
    student_id: str = Form(...),
    name: str = Form(...),
    img: Optional[UploadFile] = File(None),
    img_path: Optional[str] = Form(None),
    model_name: str = Form("Facenet"),
    detector_backend: str = Form("fastmtcnn"),
    enforce_detection: bool = Form(True),
    align: bool = Form(True)
):
    if img is None and img_path is None:
        raise HTTPException(status_code=400, detail="Either 'img' or 'img_path' must be provided.")
    
    img_input = img if img is not None else img_path
    
    return await register_student(
        student_id=student_id,
        name=name,
        img=img_input,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )