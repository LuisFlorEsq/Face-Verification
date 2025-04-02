import os
import shutil

# Define source and destination directories
source_dir = "training-images"  # Folder with all student images
destination_dir = "image-database"  # Folder with reorganized images

# Ensure destination folder exists
os.makedirs(destination_dir, exist_ok=True)

# Iterate through student ID folders
for student_id in os.listdir(source_dir):
    student_path = os.path.join(source_dir, student_id)

    if os.path.isdir(student_path):  # Ensure it's a folder
        # Create new structure
        train_path = os.path.join(destination_dir, student_id, "training")
        test_path = os.path.join(destination_dir, student_id, "testing")

        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        # Move images to respective folders
        for image in os.listdir(student_path):
            src_image_path = os.path.join(student_path, image)
            if image == "frame_15.png":
                shutil.move(src_image_path, os.path.join(train_path, image))
            else:
                shutil.move(src_image_path, os.path.join(test_path, image))

print("Files reorganized successfully!")
