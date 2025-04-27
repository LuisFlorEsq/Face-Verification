from fastapi import APIRouter

router = APIRouter(tags=["Health Check"])


@router.get("/health")
async def health_check():
    return {"status": "Face API is running!"}
