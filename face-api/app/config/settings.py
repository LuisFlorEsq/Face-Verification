import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Load environment variables from .env file
    HOST = os.getenv("HOST".replace("\r", ""))
    PORT = int(os.getenv("PORT".replace("\r", "")))
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY".replace("\r", ""))
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT".replace("\r", ""))
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME".replace("\r", ""))
    PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE".replace("\r", ""))