import pytest
from unittest.mock import patch, MagicMock
from app.services.post_register_service import register_student
from app.schemas.register_student_schema import RegisterResponse

@pytest.mark.asyncio
async def test_register_student_success():
    with patch("app.services.post_register_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_register_service.StudentRepository") as mock_repo_cls, \
         patch("app.services.post_register_service.load_image", return_value="mocked_image"), \
         patch("app.services.post_register_service.DeepFace.represent", return_value=[{"embedding": [0.1, 0.2, 0.3]}]):
        
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        mock_repo = MagicMock()
        mock_repo_cls.return_value = mock_repo
        mock_repo.student_exists.return_value = False

        response = await register_student(
            student_id="123",
            name="Luis Flores",
            img="123.jpg",
            model_name="Facenet",
            detector_backend="fastmtcnn",
            enforce_detection=True,
            align=True
        )

        assert isinstance(response, RegisterResponse)
        assert response.student_id == "123"
        assert response.message == "Student Luis Flores registered successfully."
        mock_repo.save_student_embedding.assert_called_once_with(
            student_id="123",
            embedding=[0.1, 0.2, 0.3],
            name="Luis Flores"
        )
        mock_repo.update_student_embedding.assert_not_called()

@pytest.mark.asyncio
async def test_register_student_update():
    with patch("app.services.post_register_service.get_pinecone_index") as mock_get_index, \
         patch("app.services.post_register_service.StudentRepository") as mock_repo_cls, \
         patch("app.services.post_register_service.load_image", return_value="mocked_image"), \
         patch("app.services.post_register_service.DeepFace.represent", return_value=[{"embedding": [0.1, 0.2, 0.3]}]):
        
        mock_index = MagicMock()
        mock_get_index.return_value = mock_index
        mock_repo = MagicMock()
        mock_repo_cls.return_value = mock_repo
        mock_repo.student_exists.return_value = True

        response = await register_student(
            student_id="123",
            name="Luis Flores",
            img="123.jpg"
        )

        assert isinstance(response, RegisterResponse)
        assert response.student_id == "123"
        assert response.message == "Student Luis Flores updated successfully."
        mock_repo.update_student_embedding.assert_called_once_with(
            student_id="123",
            embedding=[0.1, 0.2, 0.3]
        )
        mock_repo.save_student_embedding.assert_not_called()