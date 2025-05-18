import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from main import app

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.mark.asyncio
async def test_search_verify_student_endpoint_success(test_client):
    # Mock dependencies
    with patch("app.services.post_search_verify_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_search_verify_service.load_image") as mock_load_image, \
         patch("app.services.post_search_verify_service.verify") as mock_verify:

        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        mock_index.fetch.return_value = MagicMock(vectors={"test_id": {"values": [0.1, 0.2, 0.3]}})
        mock_load_image.return_value = "mocked_image"
        mock_verify.return_value = {"verified": True, "distance": 0.5, "threshold": 0.7, "model": "Facenet", "detector_backend": "fastmtcnn", "similarity_metric": "cosine"}

        # Make a request to the endpoint
        response = test_client.post(
            "/api/search_verify_student",
            data={"student_id": "test_id", "model_name": "Facenet", "detector_backend": "fastmtcnn", "distance_metric": "cosine", "enforce_detection": True, "align": True},
            files={"img": ("test.jpg", b"test image content")}
        )

        # Assert the response
        assert response.status_code == 200
        assert response.json()["verified"] is True
        assert response.json()["distance"] == 0.5

@pytest.mark.asyncio
async def test_search_verify_student_endpoint_not_found(test_client):
    # Mock dependencies
    with patch("app.services.post_search_verify_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_search_verify_service.load_image"):

        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        mock_index.fetch.return_value = MagicMock(vectors={})  # Student not found

        # Make a request to the endpoint
        response = test_client.post(
            "/api/search_verify_student",
            data={"student_id": "nonexistent_id"},
            files={"img": ("test.jpg", b"test image content")}
        )

        # Assert the response
        assert response.status_code == 404
        assert response.json()["detail"] == "Student ID not found."
        
@pytest.mark.asyncio
async def test_search_verify_student_endpoint_verification_error(test_client):
    # Mock dependencies
    with patch("app.services.post_search_verify_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_search_verify_service.load_image") as mock_load_image, \
         patch("app.services.post_search_verify_service.verify") as mock_verify:

        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        mock_index.fetch.return_value = MagicMock(vectors={"test_id": {"values": [0.1, 0.2, 0.3]}})
        mock_load_image.return_value = "mocked_image"
        mock_verify.side_effect = ValueError("Verification failed")

        # Make a request to the endpoint
        response = test_client.post(
            "/api/search_verify_student",
            data={"student_id": "test_id"},
            files={"img": ("test.jpg", b"test image content")}
        )

        # Assert the response
        assert response.status_code == 400
        assert "Verification failed" in response.json()["detail"]

@pytest.mark.asyncio
async def test_search_verify_student_endpoint_invalid_image(test_client):
    # Mock dependencies
    with patch("app.services.post_search_verify_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_search_verify_service.load_image") as mock_load_image, \
         patch("app.services.post_search_verify_service.verify") as mock_verify:

        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        mock_index.fetch.return_value = MagicMock(vectors={"test_id": {"values": [0.1, 0.2, 0.3]}})
        mock_load_image.side_effect = Exception("Invalid image format")

        # Make a request to the endpoint
        response = test_client.post(
            "/api/search_verify_student",
            data={"student_id": "test_id"},
            files={"img": ("test.jpg", b"test image content")}
        )

        # Assert the response
        assert response.status_code == 400
        assert "Exception while verifying" in response.json()["detail"]