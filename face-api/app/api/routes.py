from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from typing import Optional, List, Literal, Union
from pydantic import BaseModel

import numpy as np
from app.services.face_services import represent_student, register_student, search_verify_student, verify_student


router = APIRouter()

class VerifyRequest(BaseModel):
    reference: Union[str, List[float], dict, list]
    test: Union[str, List[float], dict, list]
    model_name: str = "VGG-Face"
    detector_backend: str = "opencv"
    distance_metric: str = "cosine"
    enforce_detection: bool = True
    align: bool = True

class RegisterRequest(BaseModel):
    student_id: str
    name: str
    img: Optional[UploadFile] = None
    img_path: Optional[str] = None
    model_name: str = "Facenet"
    detector_backend: str = "ssd"
    enforce_detection: bool = True
    align: bool = True
    
    

@router.post("/represent_student")
async def face_represent_student(
    img: Optional[UploadFile] = File(None),
    img_path: Optional[str] = Form(None),
    model_name: str = Form("Facenet"),
    detector_backend: str = Form("ssd"),
    enforce_detection: bool = Form(True),
    align: bool = Form(True),
    anti_spoofing: bool = Form(False),
    max_faces: Optional[int] = Form(None),
):
    
    img_input = img if img is not None else img_path
    
    if img_input is None:
        
        raise HTTPException(status_code=400, detail="Either 'img' or 'img_path' must be provided.")
    
    return represent_student(
        img=img_input,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces,
    )

@router.post("/register_student")
async def face_register_student(
    student_id: str = Form(...),
    name: str = Form(...),
    img: Optional[UploadFile] = File(None),
    img_path: Optional[str] = Form(None),
    model_name: str = Form("Facenet"),
    detector_backend: str = Form("ssd"),
    enforce_detection: bool = Form(True),
    align: bool = Form(True)
):
    
    # Check if either img or img_path is provided
    if img is None and img_path is None:
        raise HTTPException(status_code=400, detail="Either 'img' or 'img_path' must be provided.")
    
    # Img input can be either an UploadFile or a file path
    img_input = img if img is not None else img_path
    
    return register_student(
        student_id=student_id,
        name=name,
        img=img_input,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align
    )
    
@router.post("/search_verify_student")
async def face_search_verify_student(
    student_id: str = Form(...),
    img: Optional[UploadFile] = File(None),
    img_path: Optional[str] = Form(None),
    model_name: str = Form("Facenet"),
    detector_backend: str = Form("ssd"),
    distance_metric: str = Form("cosine"),
    enforce_detection: bool = Form(True),
    align: bool = Form(True) 
):
    
    reference_img = img if img is not None else img_path
    
    if reference_img is None:
        raise HTTPException(status_code=400, detail="Either 'img' or 'img_path' must be provided.")
    
    return search_verify_student(
        student_id=student_id,
        reference_img=reference_img,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align
    )

@router.post("/verify_student")
async def face_verify_student(
    request: VerifyRequest,
):
    try:
        # print(f"[DEBUG] Received request: {request}")
        
        result = verify_student(
            reference=request.reference,
            test=request.test,
            model_name=request.model_name,
            detector_backend=request.detector_backend,
            distance_metric=request.distance_metric,
            enforce_detection=request.enforce_detection,
            align=request.align
        )
        
        return result
    
    except Exception as e:
        
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/verify_student_images")
async def face_verify_student_images(
    reference: Optional[UploadFile] = File(None),
    reference_path: Optional[str] = Form(None),
    test: Optional[UploadFile] = File(None),
    test_path: Optional[str] = Form(None),
    model_name: str = Form("VGG-Face"),
    detector_backend: str = Form("opencv"),
    distance_metric: str = Form("cosine"),
    enforce_detection: bool = Form(True),
    align: bool = Form(True),
):
    try:
        reference_input = reference if reference is not None else reference_path
        test_input = test if test is not None else test_path
        if reference_input is None or test_input is None:
            raise HTTPException(status_code=400, detail="Both reference and test inputs must be provided.")

        result = verify_student(
            reference=reference_input,
            test=test_input,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align
        )
        
        return result
    
    except Exception as e:
        
        raise HTTPException(status_code=400, detail=str(e))