import os
import pickle
from deepface import DeepFace

EMBEDDINGS_FILE = "embeddings.pkl"
MODEL_NAME = "Facenet"
IMAGE_DATABASE_PATH = "image-database"

def is_image(file):
    
    """Check if a file is an image based on its extension."""
    
    return file.lower().endswith((".jpg", ".jpeg", ".png"))

def generate_embeddings():
    
    student_embeddings = {}

    # Iterate over student folders
    for student_id in os.listdir(IMAGE_DATABASE_PATH):
        student_folder = os.path.join(IMAGE_DATABASE_PATH, student_id)

        # Ensure it's a directory
        if not os.path.isdir(student_folder):
            continue  

        print(f"Processing student: {student_id}")

        # Find images **only in the main student folder** (not nested)
        images = [img for img in os.listdir(student_folder) if is_image(img)]

        if not images:
            print(f"⚠️ No images found for {student_id}, skipping...")
            continue  

        # Use the first image found as the reference
        reference_image = os.path.join(student_folder, images[0])
        print(f"→ Using {reference_image} as reference image.")

        # Generate embedding
        try:
            embedding = DeepFace.represent(img_path=reference_image, model_name=MODEL_NAME)[0]["embedding"]
            student_embeddings[student_id] = embedding
        except Exception as e:
            print(f"❌ Error processing {reference_image}: {e}")

    # Save embeddings to file
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(student_embeddings, f)
    print(f"✅ Embeddings saved to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    
    generate_embeddings()
