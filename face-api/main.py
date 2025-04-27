import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import tensorflow as tf

from app.utils.setup import configure_middleware, include_routers
from app.config.settings import Config

app = FastAPI(title="Face verification API")

configure_middleware(app)
include_routers(app)


if __name__ == "__main__":

    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT
    )