import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Load environment variables from .env file
    
    # General config
    HOST = os.getenv("HOST".replace("\r", ""))
    PORT = int(os.getenv("PORT".replace("\r", "")))
    
    # Pinecone dependecies
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY".replace("\r", ""))
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT".replace("\r", ""))
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME".replace("\r", ""))
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE".replace("\r", ""))
    
    
    # Face Recognition Settings
    DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "Facenet")
    DEFAULT_DETECTOR_BACKEND = os.getenv("DEFAULT_DETECTOR_BACKEND", "fastmtcnn")
    DEFAULT_DISTANCE_METRIC = os.getenv("DEFAULT_DISTANCE_METRIC", "cosine")
    ENFORCE_DETECTION = os.getenv("ENFORCE_DETECTION", "True").lower() == "true"
    ALIGN_FACES = os.getenv("ALIGN_FACES", "True").lower() == "true"