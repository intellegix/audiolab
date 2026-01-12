"""
AudioLab Tracks API Routes
REST endpoints for track management
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_tracks():
    """List tracks - TODO"""
    return {"message": "List tracks - TODO"}