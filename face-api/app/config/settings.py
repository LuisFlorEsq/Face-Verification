import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Load environment variables from .env file
    HOST = os.getenv("HOST".replace("\r", ""))
    PORT = int(os.getenv("PORT".replace("\r", "")))