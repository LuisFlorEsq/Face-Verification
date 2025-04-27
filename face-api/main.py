import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
import os

from app.config.settings import Config

app = FastAPI(title="Face verification API")

# Adding CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Include the router with the API routes
app.include_router(router)


if __name__ == "__main__":

    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT
    )