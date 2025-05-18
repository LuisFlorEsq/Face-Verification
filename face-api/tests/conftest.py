import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

from main import app
from app.repositories.student_repository import StudentRepository

@pytest.fixture
def test_client():
    return TestClient(app)

@pytest.fixture
def mock_pinecone_index():
    """Fixture for a mock Pinecone index."""
    return MagicMock()

@pytest.fixture
def mock_student_repository(mock_pinecone_index):
    """Fixture for a mock StudentRepository."""
    mock_repo = MagicMock(spec=StudentRepository)
    mock_repo.index = mock_pinecone_index
    return mock_repo


@pytest.fixture
def mock_face_detector():
    with patch("app.utils.verification.detection.extract_faces") as mock:
        yield mock

@pytest.fixture
def mock_model():
    with patch("app.utils.verification.modeling.build_model") as mock:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.2, 0.3]])
        mock.return_value = mock_model
        yield mock