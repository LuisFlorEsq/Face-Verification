import numpy as np
from typing import Union

def cosine_distance(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    dot_product = np.dot(embedding1, embedding2)
    
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
        
    similarity = dot_product / (norm1 * norm2)
    
    return float(similarity)

def euclidean_distance(embedding1, embedding2):
    """Compute euclidean distance between two embeddings."""
    return float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))

def l2_normalize(x):
    """L2 normalize the input vector."""
    return x / np.linalg.norm(x)