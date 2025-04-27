from typing import Union, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.schemas.search_verify_student_schema import VerifyRequest

from app.schemas.search_verify_student_schema import VerifyResponse
from app.services.post_search_verify_service import search_verify_student

router = APIRouter(tags=["Search Verify Student"])

@router.post("/search_verify_student", response_model=VerifyResponse)
async def face_search_verify_student(
    student_id: str = Form(...),
    img: Optional[UploadFile] = File(None),
    img_path: Optional[str] = Form(None),
    model_name: str = Form("Facenet"),
    detector_backend: str = Form("fastmtcnn"),
    distance_metric: str = Form("cosine"),
    enforce_detection: bool = Form(True),
    align: bool = Form(True)
):
    
    reference_img = img if img is not None else img_path
    
    if reference_img is None:
        raise HTTPException(status_code=400, detail="Either 'img' or 'img_path' must be provided.")
    
    return await search_verify_student(
        student_id=student_id,
        reference_img=reference_img,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align,
    )