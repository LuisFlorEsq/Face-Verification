import numpy as np
from typing import Union

def find_cosine_distance(source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]) -> Union[np.float64, np.ndarray]:
    """
    Find cosine distance between two given vectors or batches of vectors.
    Args:
        source_representation (np.ndarray or list): 1st vector or batch of vectors.
        test_representation (np.ndarray or list): 2nd vector or batch of vectors.
    Returns
        np.float64 or np.ndarray: Calculated cosine distance(s).
        It returns a np.float64 for single embeddings and np.ndarray for batch embeddings.
    """
    
    # Convert input to numpy array if necessary
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)
    
    if source_representation.ndim == 1 and test_representation.ndim == 1:
        
        # Single embedding
        dot_product = np.dot(source_representation, test_representation)
        source_norm = np.linalg.norm(source_representation)
        test_norm = np.linalg.norm(test_representation)
        distances = 1 - dot_product / (source_norm * test_norm)
    
    else:
        
        raise ValueError(
            f"Embeddings must be 1D or 2D, but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}")
    
    return distances


def find_euclidean_distance(source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]) -> Union[np.float64, np.ndarray]:
    """
    Find euclidean distance between two given vectors or batches of vectors.

    Args:
        source_representation (Union[np.ndarray, list]): 1st vector or batch of vectors.
        test_representation (Union[np.ndarray, list]): 1st vector or batch of vectors.

    Returns:
        np.float64 or np.ndarray: Euclidean distance(s).
            Returns np.float64 for single embeddings and np.ndarray for batch embeddings.
    """
    
    # Convert inputs to numpy arrays if necessary
    source_representation = np.asarray(source_representation)
    test_representation = np.asarray(test_representation)

    # Single embedding case (1D arrays)
    if source_representation.ndim == 1 and test_representation.ndim == 1:
        distances = np.linalg.norm(source_representation - test_representation)
    
    else: 
        
        raise ValueError(
            f"Embeddings must be 1D or 2D but received "
            f"source shape: {source_representation.shape}, test shape: {test_representation.shape}"
        )
    
    return distances
    
    
def l2_normalize(x: Union[np.ndarray, list], axis: Union[int, None] = None, epsilon: float = 1e-10) -> np.ndarray:
    """
    Normalize input vector with l2
    Args:
        x (np.ndarray or list): given vector
        axis (int): axis along which to normalize
    Returns:
        np.ndarray: l2 normalized vector
    """
    
    # Convert inputs to numpy arrays if necessary
    x = np.asarray(x)
    
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    
    return x / (norm + epsilon)


def find_distance(
    reference_embedding: Union[np.ndarray, list], 
    test_embedding: Union[np.ndarray, list],
    distance_metric: str = "euclidean",
    ) -> Union[np.float64, np.ndarray]:
    """
    Wrapper to find the distance between vectors based on the specified distance metric.

    Args:
        reference_embedding (np.ndarray or list): 1st vector or batch of vectors.
        test_embedding (np.ndarray or list): 2nd vector or batch of vectors.
        distance_metric (str): The type of distance to compute
            ('cosine', 'euclidean', 'euclidean_l2', or 'angular').

    Returns:
        np.float64 or np.ndarray: The calculated distance(s).
    """
    
    # Convert input to numpy array if necessary
    
    reference_embedding = np.asarray(reference_embedding)
    test_embedding = np.asarray(test_embedding)
    
    # Ensure that both embeddings are either 1D or 2D
    if reference_embedding.ndim != test_embedding.ndim or reference_embedding.ndim not in (1, 2):
        raise ValueError(
            f"Both embeddings must be either 1D or 2D, but received "
            f"alpha shape: {reference_embedding.shape}, beta shape: {test_embedding.shape}"
        )
        
    if distance_metric == "cosine":
        
        distance = find_cosine_distance(reference_embedding, test_embedding)
    
    elif distance_metric == "euclidean":
        
        distance = find_euclidean_distance(reference_embedding, test_embedding)
        
    elif distance_metric == "euclidean_l2":
        
        axis = None if reference_embedding.ndim == 1 else 1
        
        normalized_reference = l2_normalize(reference_embedding, axis=axis)
        normalized_test = l2_normalize(test_embedding, axis=axis)
            
        distance = find_euclidean_distance(normalized_reference, normalized_test)
        
    else: 
        
        raise ValueError(f"Unsupported distance metric: {distance_metric}. Supported metrics are: 'cosine', 'euclidean', 'euclidean_l2")
        
    
    # Round the distance to 6 decimal places
    
    final_distance = np.round(distance, 6)
    
    return final_distance


def find_threshold(model_name: str, distance_metric: str) -> float:
    """
    Receive pre-tuned threshold values for the given model and distance metric.

    Args:
        model_name (str): Model for face recognition.
        distance_metric (str): Distance metric used to calculate the similarity.

    Returns:
        threshold (float): Threshold value for the model and distance metric pair. 
    """
    
    base_threshold = {"cosine": 0.40, "euclidean": 0.55, "euclidean_l2": 0.75, "angular": 0.37}

    thresholds = {
        "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
        "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
        "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
    }

    threshold = thresholds.get(model_name, base_threshold).get(distance_metric, 0.4)
    
    return threshold
