from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from typing import Optional, List, Literal

from app.services.face_services import represent, verify
from app.utils.similarity import cosine_distance, euclidean_distance

router = APIRouter()

@router.post("/represent")
async def face_represent(
    img: Optional[UploadFile] = File(None),
    img_path: Optional[str] = Form(None),
    model_name: str = Form("VGG-Face"),
    detector_backend: str = Form("opencv"),
    enforce_detection: bool = Form(True),
    align: bool = Form(True),
    anti_spoofing: bool = Form(False),
    max_faces: Optional[int] = Form(None),
):
    
    img_input = img if img is not None else img_path
    
    if img_input is None:
        
        raise HTTPException(status_code=400, detail="Either 'img' or 'img_path' must be provided.")
    
    return represent(
        img=img_input,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces,
    )

@router.post("/compare-embeddings")
async def compare_embeddings(
    embedding1: List[float] = Body(..., embed=True),
    embedding2: List[float] = Body(..., embed=True),
    metric: Literal["cosine", "euclidean", "manhattan"] = Body("cosine", embed=True)
):
    if len(embedding1) != len(embedding2):
        raise HTTPException(status_code=400, detail="Embeddings must be the same length")

    try:
        if metric == "cosine":
            score = cosine_distance(embedding1, embedding2)
        elif metric == "euclidean":
            score = euclidean_distance(embedding1, embedding2)
        else:
            raise HTTPException(status_code=400, detail="Invalid metric")

        return {"metric": metric, "score": score}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/verify")
async def face_verify(
    img1: Optional[UploadFile] = File(None),
    img2: Optional[UploadFile] = File(None),
    img1_path: Optional[str] = Form(None),
    img2_path: Optional[str] = Form(None),
    model_name: str = Form("VGG-Face"),
    detector_backend: str = Form("opencv"),
    distance_metric: str = Form("cosine"),
    enforce_detection: bool = Form(True),
    align: bool = Form(True),
    anti_spoofing: bool = Form(False),
):
    return verify(
        img1=img1 or img1_path,
        img2=img2 or img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align,
        anti_spoofing=anti_spoofing,
    )
