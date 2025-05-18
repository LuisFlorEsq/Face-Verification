# import pytest
# import numpy as np
# import cv2
# from fastapi import HTTPException, UploadFile
# from unittest.mock import MagicMock
# from app.utils.process_image import load_image
# from io import BytesIO

# def create_sample_image():
#     """Creates a sample BGR image as a NumPy array."""
#     return np.zeros((100, 100, 3), dtype=np.uint8)

# def create_sample_upload_file(image_bytes: bytes, filename: str = "test.jpg"):
#     """Creates a sample UploadFile object from image bytes."""
#     file = MagicMock()
#     file.filename = filename
#     # Create a BytesIO object with the image bytes
#     file.file = BytesIO(image_bytes)
#     return file

# def encode_image(image: np.ndarray, ext: str = ".jpg") -> bytes:
#     """Encodes a NumPy array image to bytes."""
#     _, encoded_image = cv2.imencode(ext, image)
#     return encoded_image.tobytes()

# @pytest.mark.asyncio
# async def test_load_image_from_file_path():
#     # Arrange
#     sample_image = create_sample_image()
#     image_bytes = encode_image(sample_image)
#     file_path = "test_image.jpg"
#     with open(file_path, "wb") as f:
#         f.write(image_bytes)

#     try:
#         # Act
#         loaded_image = load_image(file_path)  # Remove await since it's not async

#         # Assert
#         assert isinstance(loaded_image, np.ndarray)
#         assert loaded_image.shape == (100, 100, 3)
#         assert loaded_image.dtype == np.uint8
#     finally:
#         # Cleanup
#         import os
#         os.remove(file_path)

# @pytest.mark.asyncio
# async def test_load_image_from_upload_file():
#     # Arrange
#     sample_image = create_sample_image()
#     image_bytes = encode_image(sample_image)
#     upload_file = create_sample_upload_file(image_bytes)

#     # Act
#     loaded_image = load_image(upload_file)  # Remove await

#     # Assert
#     assert isinstance(loaded_image, np.ndarray)
#     assert loaded_image.shape == (100, 100, 3)
#     assert loaded_image.dtype == np.uint8

# def test_load_image_invalid_file_path():
#     # Act & Assert
#     with pytest.raises(HTTPException) as exc_info:
#         load_image("non_existent_file.jpg")  # Remove await
#     assert exc_info.value.status_code == 400
#     assert "Failed to load image!" in exc_info.value.detail

# def test_load_image_invalid_format():
#     # Arrange
#     invalid_file = create_sample_upload_file(b"invalid image data", "test.txt")

#     # Act & Assert
#     with pytest.raises(HTTPException) as exc_info:
#         load_image(invalid_file)  # Remove await
#     assert exc_info.value.status_code == 400
#     assert "Failed to load image!" in exc_info.value.detail