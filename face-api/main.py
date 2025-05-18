import uvicorn
from fastapi import FastAPI

from app.utils.setup import configure_middleware, include_routers
from app.config.settings import Config

app = FastAPI(
    
    title = "Representación y verificación de rostros de alumnos",
    description = "API para la representación de rostros como embeddings y verificación de la identidad de los alumnos",
    version = "1.5.0",
    contact={
        "name": "Luis Antonio Flores Esquivel",
        "email": "flores.esquivel.luis.antonio2@gmail.com"
    }
    
)

configure_middleware(app)
include_routers(app)


if __name__ == "__main__":

    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT
    )