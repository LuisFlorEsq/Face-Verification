from fastapi import HTTPException, UploadFile
from typing import Union, Optional

from deepface import DeepFace

from app.utils.process_image import load_image
from app.schemas.represent_student_schema import RepresentResponse

async def represent_student(
    img: Union[UploadFile, str],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    anti_spoofing: bool = False,
    max_faces: Optional[int] = None
) -> RepresentResponse:
    """
    Get the face embeddings for the given image.

    Args:
        img (Union[UploadFile, str]): Image can be a file path, URL or UploadFile object.
        model_name (str, optional): The face recognition model to generate embeddings. Defaults to "VGG-Face".
        detector_backend (str, optional): Face detection model. Defaults to "opencv".
        enforce_detection (bool, optional): Perform face detection before obtain embedding. Defaults to True.
        align (bool, optional): Align the detected face. Defaults to True.

    Returns:
        results (List[Dict[str, Any]] or List[Dict[str, Any]]): A list of dictionaries.
        Result type becomes List of List of Dict if batch input passed.
    """
    
    
    try:
        image = load_image(img)
        embedding_objs = DeepFace.represent(
            img_path=image,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        
        return RepresentResponse(results=embedding_objs)
    
    except Exception as err:
        import traceback
        tb_str = traceback.format_exc()
        raise HTTPException(status_code=400, detail=f"Exception while representing: {str(err)}\n{tb_str}")
