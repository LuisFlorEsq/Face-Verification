import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException
from app.services.post_register_service import register_student
from app.schemas.register_student_schema import RegisterResponse

# Mock FetchResponse and Vector classes
class MockVector:
    def __init__(self, id, values, metadata):
        self.id = id
        self.values = values
        self.metadata = metadata

class MockFetchResponse:
    def __init__(self, vectors):
        self.vectors = vectors
        self.namespace = ''
        self.usage = {'read_units': 1}

@pytest.mark.asyncio
async def test_register_student_success():
    # Mock dependencies
    mock_index = MagicMock()
    mock_index.fetch.return_value = MockFetchResponse(vectors={})
    mock_index.upsert = MagicMock()

    with patch("app.services.post_register_service.get_pinecone_index", return_value=mock_index):
        with patch("app.services.post_register_service.load_image", return_value="mocked_image"):
            with patch(
                "app.services.post_register_service.DeepFace.represent",
                return_value=[{"embedding": [0.1, 0.2, 0.3]}]
            ):
                # Call the service
                response = await register_student(
                    student_id="123",
                    name="Luis Flores",
                    img="123.jpg",
                    model_name="Facenet",
                    detector_backend="fastmtcnn",
                    enforce_detection=True,
                    align=True
                )

                # Assertions
                assert isinstance(response, RegisterResponse)
                assert response.student_id == "123"
                assert response.message == "Student Luis Flores registered successfully."
                mock_index.upsert.assert_called_once_with(
                    vectors=[
                        {
                            "id": "123",
                            "values": [0.1, 0.2, 0.3],
                            "metadata": {"student_id": "123", "name": "Luis Flores"}
                        }
                    ]
                )

@pytest.mark.asyncio
async def test_register_student_already_exists():
    # Mock dependencies
    mock_index = MagicMock()
    mock_index.fetch.return_value = MockFetchResponse(
        vectors={"123": MockVector(id="123", values=[0.1, 0.2, 0.3], metadata={"student_id": "123", "name": "Luis Flores"})}
    )

    with patch("app.services.post_register_service.get_pinecone_index", return_value=mock_index):
        # Call the service and expect an HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await register_student(
                student_id="123",
                name="Luis Flores",
                img="123.jpg"
            )

        # Assertions
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Student ID already exists."