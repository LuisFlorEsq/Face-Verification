import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# DeepFace module
from deepface import DeepFace


# Define paths
training_root = "./image-database"  # Folder with all student images
faces_output = "./extracted_faces"  # Folder to save extracted faces
os.makedirs(faces_output, exist_ok=True)

# Iterate through student folders
for student_id in os.listdir(training_root):
    training_path = os.path.join(training_root, student_id, "training")

    if os.path.isdir(training_path):
        for image_name in os.listdir(training_path):
            image_path = os.path.join(training_path, image_name)

            try:
                # Extract faces
                faces = DeepFace.extract_faces(img_path=image_path, align=True, detector_backend="mtcnn", enforce_detection=True)

                if faces:
                    # Process extracted faces
                    for i, face_data in enumerate(faces):
                        face_array = face_data["face"]  # NumPy array of the face
                        save_path = os.path.join(faces_output, f"{student_id}_face_{i}.png")
                        
                        # Convert float64 (0-1) to uint8 (0-255)
                        face_array = (face_array * 255).astype(np.uint8)

                        # Save the face
                        cv2.imwrite(save_path, cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR))
                        print(f"Saved: {save_path}")

            except Exception as e:
                print(f"Error processing {image_path}: {e}")