import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from app.services.post_search_verify_service import search_verify_student
from app.schemas.search_verify_student_schema import VerifyResponse

@pytest.mark.asyncio
async def test_search_verify_student_success():
    # Mock dependencies
    mock_repo = MagicMock()
    mock_repo.get_student_embedding.return_value = [0.1, 0.2, 0.3]  # Mock embedding

    with patch("app.services.post_search_verify_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_search_verify_service.StudentRepository", return_value=mock_repo), \
         patch("app.services.post_search_verify_service.load_image", return_value="mocked_image"), \
         patch("app.services.post_search_verify_service.verify") as mock_verify_patch:

        mock_verify_patch.return_value = {"verified": True, "distance": 0.5, "threshold": 0.7, "model": "Facenet", "detector_backend": "fastmtcnn", "similarity_metric": "cosine"}

        response = await search_verify_student(
            student_id="123",
            reference_img="test.jpg",
            model_name="Facenet",
            detector_backend="fastmtcnn",
            distance_metric="cosine",
            enforce_detection=True,
            align=True
        )
        
        # Assertions
        assert isinstance(response, VerifyResponse)
        assert response.verified is True
        assert response.distance == 0.5

@pytest.mark.asyncio
async def test_search_verify_student_not_found():
    # Mock dependencies
    mock_repo = MagicMock()
    mock_repo.get_student_embedding.return_value = None  # Student not found

    with patch("app.services.post_search_verify_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_search_verify_service.StudentRepository", return_value=mock_repo), \
         patch("app.services.post_search_verify_service.load_image", return_value="mocked_image"):

        # Call the service and expect an HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await search_verify_student(
                student_id="123",
                reference_img="test.jpg",
                model_name="Facenet",
                detector_backend="fastmtcnn",
                distance_metric="cosine",
                enforce_detection=True,
                align=True
            )

        # Assertions
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Student ID not found."

@pytest.mark.asyncio
async def test_search_verify_student_verification_error():
    # Mock dependencies
    mock_repo = MagicMock()
    mock_repo.get_student_embedding.return_value = [0.1, 0.2, 0.3]  # Mock embedding

    with patch("app.services.post_search_verify_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_search_verify_service.StudentRepository", return_value=mock_repo), \
         patch("app.services.post_search_verify_service.load_image", return_value="mocked_image"), \
         patch("app.services.post_search_verify_service.verify") as mock_verify_patch:

        mock_verify_patch.side_effect = ValueError("Verification failed")

        # Call the service and expect an HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await search_verify_student(
                student_id="123",
                reference_img="test.jpg",
                model_name="Facenet",
                detector_backend="fastmtcnn",
                distance_metric="cosine",
                enforce_detection=True,
                align=True
            )

        # Assertions
        assert exc_info.value.status_code == 400
        assert "Verification failed" in exc_info.value.detail