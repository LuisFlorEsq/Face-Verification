from typing import Union, Dict, Any
from fastapi import HTTPException, UploadFile
from deepface import DeepFace

from app.utils.proces_image import load_image
from app.utils.setup import get_chroma_collection
from app.schemas.register_student_schema import RegisterResponse

async def register_student(
    student_id: str,
    name: str,
    img: Union[UploadFile, str],
    model_name: str = "Facenet",
    detector_backend: str = "ssd",
    enforce_detection: bool = True,
    align: bool = True
) -> RegisterResponse:
    """
    Register a student by storing their face embedding in ChromaDB.

    Args:
        student_id (str): The current student ID to be enrolled.
        name (str): The name of the student.
        img (Union[UploadFile, str]): The reference image of the student, can be a file path, URL or UploadFile object.
        model_name (str, optional): Embedding generation model. Defaults to "Facenet".
        detector_backend (str, optional): Face detection model. Defaults to "ssd".
        enforce_detection (bool, optional): Ensure to detect a face before embedding generation. Defaults to True.
        algin (bool, optional): Align the detected face before embedding generation. Defaults to True.

    Returns:
        Dict[str, Any]: Registration status and details.
    """
    
    # # log_resources("Before register_student")
    
    try:
        
        # Check if student_id already exists in the database
        
        collection = get_chroma_collection()
        existing_student = collection.get(ids=[student_id])
        
        if existing_student["ids"]:
            raise HTTPException(status_code=400, detail="Student ID already exists.")
        
        
        # Generate embedding
        image = load_image(img)
        
        embedding_obj = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align
        )
        
        # print(f"[DEBUG]: Embedding object", embedding_obj)
        # print(f"[DEBUG]: Embedding", embedding_obj[0]["embedding"])
        
        embedding = embedding_obj[0]["embedding"]
        
        # Store the embedding in ChromaDB
        
        collection.add(
            embeddings=[embedding],
            metadatas=[{"student_id": student_id, "name": name}],
            ids=[student_id]
            
        )
        
        # # log_resources("After register_student")
        
        return RegisterResponse(
            message=f"Student {name} registered successfully.",
            student_id=student_id,
        )
        
    
    except Exception as err:
        
        import traceback
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Exception while registeing: {str(err)}\n{tb_str}")