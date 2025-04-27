import importlib
import pkgutil
from fastapi.middleware.cors import CORSMiddleware
from app.routes import __name__ as routes_module

import chromadb
from chromadb.utils import embedding_functions
import os


def configure_middleware(app):
    """
    Configure middleware for the FastAPI application.
    """
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"]
    )
    
    
def include_routers(app):
    """
    Discover and include all routers from the routes module.
    """
    
    for module_info in pkgutil.iter_modules([routes_module.replace('.', '/')]):
        
        module_name = f"{routes_module}.{module_info.name}"
        module = importlib.import_module(module_name)
        
        if hasattr(module, 'router'):
            app.include_router(module.router, prefix="/api")
            

def get_chroma_collection():
    
    CHROMA_PATH = os.getenv("CHROMA_PERSISTENT_PATH", "./students_db")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    return client.get_or_create_collection(
        name="student_embeddings",
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )