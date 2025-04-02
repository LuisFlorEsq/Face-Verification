from fastapi import FastAPI
from app.api.routes import router

app = FastAPI()

# Include the router with the API routes
app.include_router(router)

@app.get("/")
def home():
    return {"message": "Welcome to DeepFace API with FastAPI!"}
