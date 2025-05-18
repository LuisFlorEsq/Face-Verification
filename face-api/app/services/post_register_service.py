from typing import Union, Dict, Any
from fastapi import HTTPException, UploadFile
from deepface import DeepFace

from app.utils.process_image import load_image
from app.utils.setup import get_pinecone_index
from app.schemas.register_student_schema import RegisterResponse
from app.repositories.student_repository import StudentRepository

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
    Register a student by storing their face embedding in Pinecone.

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
    
    try:
        
        # Get the student repository
        index = get_pinecone_index()
        repository = StudentRepository(index)
        
        # Generate embedding
        image = load_image(img)
        
        embedding_obj = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align
        )
        
        # Obtain the embedding from the overall response
        embedding = embedding_obj[0]["embedding"]
        
        # Check if the student ID already exists in the database
        if repository.student_exists(student_id=student_id):
            
            repository.update_student_embedding(
                student_id=student_id,
                embedding=embedding
            )
            
            return RegisterResponse(
                message=f"Student {name} updated successfully.",
                student_id=student_id
            )
        
        # Save the new student embedding to the database
        repository.save_student_embedding(
            student_id=student_id, 
            embedding=embedding, 
            name = name
        )
                
                
        return RegisterResponse(
            message=f"Student {name} registered successfully.",
            student_id=student_id,
        )
        
    
    except HTTPException as http_err:
        
        raise http_err
        
    except Exception as err:
        
        import traceback
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Exception while registering: {str(err)}\n{tb_str}")