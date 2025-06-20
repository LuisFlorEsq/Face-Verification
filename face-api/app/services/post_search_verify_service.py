from typing import Union, Dict, Any
from fastapi import HTTPException, UploadFile

from app.utils.process_image import load_image
from app.utils.setup import get_pinecone_index
from app.utils.verification import verify
from app.schemas.search_verify_student_schema import VerifyResponse
from app.repositories.student_repository import StudentRepository
from app.config.settings import Config

from deepface.commons.logger import Logger


logger = Logger()

async def search_verify_student(
    student_id: str,
    reference_img: Union[UploadFile, str],
    model_name: str = Config.DEFAULT_MODEL_NAME,
    detector_backend: str = Config.DEFAULT_DETECTOR_BACKEND,
    distance_metric: str = Config.DEFAULT_DISTANCE_METRIC,
    enforce_detection: bool = Config.ENFORCE_DETECTION,
    align: bool = Config.ALIGN_FACES,
) -> VerifyResponse:
    """
    Search for a student's embedding in the database and verify it against a reference image.

    Args:
        student_id (str): The student ID to search for.
        reference_img (Union[UploadFile, str]): The reference image to verify against.
        model_name (str, optional): Embedding generation model. Defaults to "Facenet".
        detector_backend (str, optional): Face detection model. Defaults to "ssd".
        enfoce_detection (bool, optional): Ensure to detect a face before embedding generation. Defaults to True.
        align (bool, optional): Align the detected face before embedding generation. Defaults to True.

    Returns:
        Dict[str, Any]: Verification result including distance and status.
    """
        
    try:
        
        # Get reference image and stored embedding
        reference_img = load_image(reference_img)
        
        # Fetch the stored embedding from the database
        index = get_pinecone_index()
        repository = StudentRepository(index)
        
        reference_embedding = repository.get_student_embedding(student_id)
        

        # Check if the student ID exists in the database
        if reference_embedding is None:
            
            raise HTTPException(status_code=404, detail="Student ID not found.")
        
        # Verify both embeddings
        student_result = verify(
            img1_path=reference_embedding,
            img2_path=reference_img,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align  
        )
        
        return VerifyResponse(**student_result)
   
    
    except HTTPException as http_err:
        
        raise http_err
    
    except ValueError as ve:

        # Handle DeepFace-specific errors (e.g., face detection failure)
        logger.error(f"ValueError during verification for student {student_id}: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Verification failed: {str(ve)}")
        
    
    except Exception as err:
        
        import traceback
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Exception while verifying: {str(err)}\n{tb_str}")
