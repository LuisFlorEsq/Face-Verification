# import pytest
# import numpy as np
# from unittest.mock import patch, MagicMock
# from fastapi import HTTPException
# from app.utils.verification import verify
# from app.utils.similarity import find_cosine_distance, find_threshold

# @pytest.fixture
# def mock_face_detector():
#     with patch("app.utils.verification.detection.extract_faces") as mock:
#         mock.return_value = [{"face": np.zeros((100, 100, 3)), "is_real": True}]
#         yield mock

# @pytest.fixture
# def mock_model():
#     with patch("app.utils.verification.modeling.build_model") as mock:
#         mock_model = MagicMock()
#         mock_model.output_shape = 3  # Match the embedding dimension
#         mock.return_value = mock_model
#         yield mock

# @pytest.fixture
# def mock_represent():
#     with patch("app.utils.verification.representation.represent") as mock:
#         mock.return_value = [{"embedding": [0.1, 0.2, 0.3]}]
#         yield mock

# def test_verify_success(mock_face_detector, mock_model, mock_represent):
#     # Act
#     result = verify(
#         img1_path="test1.jpg",
#         img2_path="test2.jpg",
#         model_name="Facenet",
#         detector_backend="opencv",
#         distance_metric="cosine",
#         enforce_detection=True,
#         align=True
#     )

#     # Assert
#     assert isinstance(result, dict)
#     assert all(key in result for key in [
#         "verified", "distance", "threshold", "model",
#         "detector_backend", "similarity_metric", "time"
#     ])

# def test_verify_with_precomputed_embeddings(mock_model):
#     # Arrange
#     embedding1 = [0.1, 0.2, 0.3]
#     embedding2 = [0.1, 0.2, 0.31]

#     # Act
#     result = verify(
#         img1_path=embedding1,
#         img2_path=embedding2,
#         model_name="Facenet",
#         distance_metric="cosine"
#     )

#     # Assert
#     assert isinstance(result, dict)
#     distance = find_cosine_distance(embedding1, embedding2)
#     threshold = find_threshold("Facenet", "cosine")
#     assert result["distance"] == pytest.approx(distance, rel=1e-5)
#     assert result["verified"] == (distance <= threshold)

# def test_verify_no_face_detected(mock_face_detector, mock_model):
#     # Arrange
#     mock_face_detector.return_value = []
    
#     # Act
#     result = verify(
#         img1_path="test1.jpg",
#         img2_path="test2.jpg",
#         model_name="Facenet",
#         detector_backend="opencv",
#         enforce_detection=True
#     )

#     # Assert
#     assert isinstance(result, dict)
#     assert result["verified"] is False
#     assert result["distance"] == 999.0