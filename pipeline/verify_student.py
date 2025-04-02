import os
import pickle
import numpy as np
from deepface import DeepFace

EMBEDDINGS_FILE = "embeddings.pkl"
MODEL_NAME = "Facenet"
IMAGE_DATABASE_PATH = "image-database"

def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embeddings."""
    
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)  # Cosine similarity formula
    
    return similarity

def verify_student(student_id, test_image_name="frame_01.jpg"):
    """Verify a student's identity using their test image."""
    
    with open(EMBEDDINGS_FILE, "rb") as f:
        student_embeddings = pickle.load(f)

    if student_id not in student_embeddings:
        return "❌ Student ID not found."

    # Construct full path to test image
    test_image_path = os.path.join(IMAGE_DATABASE_PATH, student_id, "testing", test_image_name)

    if not os.path.exists(test_image_path):
        return f"❌ Test image not found: {test_image_path}"

    # Load embeddings
    base_embedding = student_embeddings[student_id]
    test_embedding = DeepFace.represent(img_path=test_image_path, model_name=MODEL_NAME)[0]["embedding"]

    similarity = cosine_similarity(base_embedding, test_embedding)

    if similarity > 0.7:  # Adjust threshold as needed
        return f"✅ Match! Student: {student_id} (Similarity: {similarity:.2f})"
    else:
        return f"❌ No match for Student: {student_id}  (Similarity): {similarity:.2f}"