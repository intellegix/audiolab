"""
AudioLab Effects API Routes
REST endpoints for audio effects management
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_effects():
    """List available effects - TODO"""
    return {"message": "List effects - TODO"}