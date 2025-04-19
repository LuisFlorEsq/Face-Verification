import time
from typing import Any, Dict, Optional, Union, List, Tuple

import numpy as np

from deepface.modules import representation, detection, modeling
from deepface.models.FacialRecognition import FacialRecognition
from deepface.commons.logger import Logger

from app.utils.similarity import find_distance, find_threshold

logger = Logger()


def verify(
    img1_path: Union[str, np.ndarray, List[float]],
    img2_path: Union[str, np.ndarray, List[float]],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    distance_metric: str = "cosine",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    silent: bool = False,
    threshold: Optional[float] = None,
    anti_spoofing: bool = False,
) -> Dict[str, Any]:
    """
    
    Verify if two images are of the same person or not.
    
    It converts facial images to vectors and then calculates the distance between them.

    Args:
        img1_path (Union[str, np.ndarray, List[float]]): Path to the first image or a precomputed embedding.
        img2_path (Union[str, np.ndarray, List[float]]): Path to the second image or a precomputed embedding.
        model_name (str, optional): Model used to embedding generation. Defaults to "VGG-Face".
        detector_backend (str, optional): Face detection backend. Defaults to "opencv".
        distance_metric (str, optional): Metric to measure the similarity. Defaults to "cosine".
        enforce_detection (bool, optional): When no face is detected it raises an exception. Defaults to True.
        align (bool, optional): Face alignment. Defaults to True.
        expand_percentage (int, optional): Expand detected facial area. Defaults to 0.
        normalization (str, optional): Normalize the image input before feed it to the model. Defaults to "base".
        silent (bool, optional): Suppress or allow logg messages. Defaults to False.
        threshold (Optional[float], optional): The threshold used to determine if a pair represent the same person or not. Defaults to None.
        anti_spoofing (bool, optional): Anti-spoofing validation. Defaults to False.

    Returns:
        Dict[str, Any]: A dictionary containing the verification result, distance, and other details.
    """
    
    tic = time.time()
    
    model: FacialRecognition = modeling.build_model(
        task="facial_recognition", model_name=model_name
    )
    
    dims = model.output_shape
    
    def extract_embeddings(
        img_path: Union[str, np.ndarray, List[float]], index: int
    ) -> List[List[float]]:
        """
        Extracts facial embeddings from an image or returns pre-calculated embeddings.

        Depending on the type of img_path, the function either extracts
        facial embeddings from the provided image
        (via a path or NumPy array) or verifies that the input is a list of
        pre-calculated embeddings and validates them.

        Args:
            img_path (Union[str, np.ndarray, List[float]]):
                - A string representing the file path to an image,
                - A NumPy array containing the image data,
                - Or a list of pre-calculated embedding values (of type `float`).
            index (int): An index value used in error messages and logging
            to identify the number of the image.

        Returns:
            List[List[float]]:
                A list containing lists of facial embeddings for each detected face.
        """
        
        if isinstance(img_path, list):
            # given image is already pre-calculated embedding
            if not all(isinstance(dim, (float, int)) for dim in img_path):
                raise ValueError(
                    f"When passing img{index}_path as a list,"
                    " ensure that all its items are of type float."
                )

            if silent is False:
                logger.warn(
                    f"You passed {index}-th image as pre-calculated embeddings. "
                    "Please ensure that embeddings have been calculated"
                    f" for the {model_name} model."
                )

            if len(img_path) != dims:
                raise ValueError(
                    f"embeddings of {model_name} should have {dims} dimensions,"
                    f" but {index}-th image has {len(img_path)} dimensions input"
                )

            img_embeddings = [img_path]
        else:
            try:
                img_embeddings = __extract_faces_and_embeddings(
                    img_path=img_path,
                    model_name=model_name,
                    detector_backend=detector_backend,
                    enforce_detection=enforce_detection,
                    align=align,
                    expand_percentage=expand_percentage,
                    normalization=normalization,
                    anti_spoofing=anti_spoofing,
                )
            except ValueError as err:
                raise ValueError(f"Exception while processing img{index}_path") from err
            
        return img_embeddings

    img1_embeddings = extract_embeddings(img1_path, 1)
    img2_embeddings = extract_embeddings(img2_path, 2)

    min_distance = float("inf")
    for img1_embedding in img1_embeddings:
        for img2_embedding in img2_embeddings:
            distance = find_distance(img1_embedding, img2_embedding, distance_metric)
            if distance < min_distance:
                min_distance = distance

    # find the face pair with minimum distance
    threshold = threshold or find_threshold(model_name, distance_metric)
    distance = float(min_distance)

    toc = time.time()

    resp_obj = {
        "verified": distance <= threshold,
        "distance": distance,
        "threshold": threshold,
        "model": model_name,
        "detector_backend": detector_backend,
        "similarity_metric": distance_metric,
        "time": round(toc - tic, 2),
    }

    return resp_obj


def __extract_faces_and_embeddings(
    img_path: Union[str, np.ndarray],
    model_name: str = "VGG-Face",
    detector_backend: str = "opencv",
    enforce_detection: bool = True,
    align: bool = True,
    expand_percentage: int = 0,
    normalization: str = "base",
    anti_spoofing: bool = False,
) -> List[List[float]]:
    """
    Extract facial embeddings for given image
    Returns:
        embeddings (List[List[float]]): List of embeddings for detected faces
    """
    embeddings = []

    img_objs = detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        anti_spoofing=anti_spoofing,
    )

    # find embeddings for each face
    for img_obj in img_objs:
        if anti_spoofing is True and img_obj.get("is_real", True) is False:
            raise ValueError("Spoof detected in given image.")
        img_embedding_obj = representation.represent(
            img_path=img_obj["face"],
            model_name=model_name,
            enforce_detection=enforce_detection,
            detector_backend="skip",
            align=align,
            normalization=normalization,
        )
        # already extracted face given, safe to access its 1st item
        img_embedding = img_embedding_obj[0]["embedding"]
        embeddings.append(img_embedding)

    return embeddings