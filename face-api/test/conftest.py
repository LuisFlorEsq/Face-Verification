import pytest
import pinecone
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from main import app
from dotenv import load_dotenv
import os

# Load test environment variables
load_dotenv(".env.test")

@pytest.fixture(scope="session")
def mock_pinecone_index():
    """Mock Pinecone index for testing."""
    mock_index = MagicMock()
    mock_index.fetch.return_value = {"vectors": {}}
    mock_index.upsert = AsyncMock()
    mock_index.query = AsyncMock(return_value={"matches": []})
    return mock_index

@pytest.fixture(scope="session")
def mock_deepface():
    """Mock DeepFace verify function."""
    mock_verify = MagicMock()
    mock_verify.return_value = {
        "verified": True,
        "distance": 0.2,
        "threshold": 0.4,
        "model": "Facenet",
        "detector_backend": "fastmtcnn",
        "similarity_metric": "cosine"
    }
    return mock_verify

@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)

@pytest.fixture(autouse=True)
def setup_environment():
    """Set up test environment variables."""
    os.environ["PINECONE_API_KEY"] = "test-api-key"
    os.environ["PINECONE_ENVIRONMENT"] = "test-environment"
    os.environ["PINECONE_INDEX_NAME"] = "test-student-embeddings"