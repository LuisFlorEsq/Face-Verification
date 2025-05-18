import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
async def test_register_student_endpoint_success(test_client):
    # Mock dependencies
    with patch("app.services.post_register_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_register_service.load_image") as mock_load_image, \
         patch("app.services.post_register_service.DeepFace.represent") as mock_represent:

        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        mock_index.upsert = MagicMock()
        mock_load_image.return_value = "mocked_image"
        mock_represent.return_value = [{"embedding": [0.1, 0.2, 0.3]}]

        # Make a request to the endpoint
        response = test_client.post(
            "/api/register_student",
            data={"student_id": "test_id", "name": "Test Name"},
            files={"img": ("test.jpg", b"test image content")}
        )

        # Assert the response
        assert response.status_code == 200
        assert "registered" in response.json()["message"]

        # Assert that Pinecone was called
        mock_index.upsert.assert_called_once()

@pytest.mark.asyncio
async def test_register_student_endpoint_update(test_client):
    # Mock dependencies
    with patch("app.services.post_register_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_register_service.load_image") as mock_load_image, \
         patch("app.services.post_register_service.DeepFace.represent") as mock_represent:

        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        mock_index.update = MagicMock()
        mock_index.fetch.return_value = MagicMock(vectors={"test_id": {"values": [0.1, 0.2, 0.3]}})
        mock_load_image.return_value = "mocked_image"
        mock_represent.return_value = [{"embedding": [0.4, 0.5, 0.6]}]

        # Make a request to the endpoint
        response = test_client.post(
            "/api/register_student",
            data={"student_id": "test_id", "name": "Updated Name"},
            files={"img": ("test.jpg", b"test image content")}
        )

        # Assert the response
        assert response.status_code == 200
        assert "updated" in response.json()["message"]

        # Assert that Pinecone was called
        mock_index.update.assert_called_once()

@pytest.mark.asyncio
async def test_register_student_endpoint_invalid_image(test_client):
    # Mock dependencies
    with patch("app.services.post_register_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_register_service.load_image") as mock_load_image, \
         patch("app.services.post_register_service.DeepFace.represent") as mock_represent:

        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        mock_load_image.side_effect = Exception("Invalid image format")

        # Make a request to the endpoint
        response = test_client.post(
            "/api/register_student",
            data={"student_id": "test_id", "name": "Test Name"},
            files={"img": ("test.jpg", b"test image content")}
        )

        # Assert the response
        assert response.status_code == 400
        assert "Exception while registering" in response.json()["detail"]

@pytest.mark.asyncio
async def test_register_student_endpoint_deepface_error(test_client):
    # Mock dependencies
    with patch("app.services.post_register_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_register_service.load_image") as mock_load_image, \
         patch("app.services.post_register_service.DeepFace.represent") as mock_represent:

        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        mock_load_image.return_value = "mocked_image"
        mock_represent.side_effect = Exception("DeepFace failed")

        # Make a request to the endpoint
        response = test_client.post(
            "/api/register_student",
            data={"student_id": "test_id", "name": "Test Name"},
            files={"img": ("test.jpg", b"test image content")}
        )

        # Assert the response
        assert response.status_code == 400
        assert "Exception while registering" in response.json()["detail"]