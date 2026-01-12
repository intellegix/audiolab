"""
AudioLab Projects API Routes
REST endpoints for project management
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ProjectCreateRequest(BaseModel):
    """Request model for creating a project"""
    name: str
    sample_rate: int = 48000
    bit_depth: int = 24


@router.post("/")
async def create_project(request: ProjectCreateRequest):
    """Create new audio project"""
    return {"message": "Project creation endpoint - TODO"}


@router.get("/{project_id}")
async def get_project(project_id: str):
    """Get project by ID"""
    return {"message": f"Get project {project_id} - TODO"}


@router.get("/")
async def list_projects():
    """List all projects"""
    return {"message": "List projects - TODO"}