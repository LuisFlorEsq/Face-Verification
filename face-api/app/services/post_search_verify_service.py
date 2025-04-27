from typing import Union, Dict, Any
from fastapi import HTTPException, UploadFile

from app.utils.proces_image import load_image
from app.utils.setup import get_chroma_collection
from app.utils.verification import verify
from app.schemas.search_verify_student_schema import VerifyResponse


async def search_verify_student(
    student_id: str,
    reference_img: Union[UploadFile, str],
    model_name: str = "Facenet",
    detector_backend: str = "ssd",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
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
    
    # # log_resources("Before search_verify_student")
    
    try:
        
        # Get reference image and stored embedding
        reference_img = load_image(reference_img)
        
        # Query ChromaDB for the stored embedding of the student
        collection = get_chroma_collection()
        
        results = collection.get(
            ids=[student_id],
            include=["metadatas","embeddings"]
        )
        
        # print(f"[DEBUG] Search results: {results}")
        
        if not results["ids"]:
            
            raise HTTPException(status_code=404, detail="Student ID not found.")
        
        reference_embedding = results["embeddings"][0].tolist()
        
        # print(f"[DEBUG] Reference embedding: {reference_embedding}")
        # print(f"[DEBUG] Format type: {type(reference_embedding)}")
        
        
        # # log_resources("After search_verify_student")
        
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
        
    
    except Exception as err:
        
        import traceback
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Exception while verifying: {str(err)}\n{tb_str}")
