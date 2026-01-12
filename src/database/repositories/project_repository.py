"""
Project Repository
Specialized repository for Project model operations
"""

import uuid
from typing import List, Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import Project
from ..schemas import ProjectCreate, ProjectUpdate
from .base import BaseRepository, RepositoryError


class ProjectRepository(BaseRepository[Project, ProjectCreate, ProjectUpdate]):
    """Repository for Project operations"""

    def __init__(self, session: AsyncSession):
        super().__init__(Project, session)

    async def get_by_user(
        self,
        user_id: uuid.UUID,
        skip: int = 0,
        limit: int = 100
    ) -> List[Project]:
        """Get all projects for a user"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(self.model.user_id == user_id)
                .order_by(self.model.updated_at.desc())
                .offset(skip)
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting user projects: {str(e)}")

    async def get_with_tracks(self, project_id: uuid.UUID) -> Optional[Project]:
        """Get project with all tracks loaded"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(self.model.id == project_id)
                .options(
                    selectinload(self.model.tracks).selectinload(Track.clips),
                    selectinload(self.model.tracks).selectinload(Track.effects)
                )
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Error getting project with tracks: {str(e)}")

    async def count_by_user(self, user_id: uuid.UUID) -> int:
        """Count projects for a user"""
        try:
            result = await self.session.execute(
                select(func.count(self.model.id))
                .where(self.model.user_id == user_id)
            )
            return result.scalar()
        except Exception as e:
            raise RepositoryError(f"Error counting user projects: {str(e)}")

    async def search_by_user(
        self,
        user_id: uuid.UUID,
        search_term: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Project]:
        """Search projects by name for a specific user"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(
                    (self.model.user_id == user_id) &
                    (self.model.name.ilike(f"%{search_term}%"))
                )
                .order_by(self.model.updated_at.desc())
                .offset(skip)
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error searching user projects: {str(e)}")

    async def get_recent_by_user(
        self,
        user_id: uuid.UUID,
        limit: int = 10
    ) -> List[Project]:
        """Get most recently updated projects for a user"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(self.model.user_id == user_id)
                .order_by(self.model.updated_at.desc())
                .limit(limit)
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting recent projects: {str(e)}")

    async def check_ownership(self, project_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        """Check if user owns the project"""
        try:
            result = await self.session.execute(
                select(func.count(self.model.id))
                .where(
                    (self.model.id == project_id) &
                    (self.model.user_id == user_id)
                )
            )
            return result.scalar() > 0
        except Exception as e:
            raise RepositoryError(f"Error checking project ownership: {str(e)}")

    async def update_last_modified(self, project_id: uuid.UUID) -> bool:
        """Update project's updated_at timestamp"""
        try:
            project = await self.get_or_404(project_id)
            project.updated_at = func.now()
            await self.session.flush()
            return True
        except Exception as e:
            raise RepositoryError(f"Error updating project timestamp: {str(e)}")

    async def get_by_sample_rate(
        self,
        sample_rate: int,
        user_id: Optional[uuid.UUID] = None
    ) -> List[Project]:
        """Get projects by sample rate, optionally filtered by user"""
        try:
            query = select(self.model).where(self.model.sample_rate == sample_rate)

            if user_id:
                query = query.where(self.model.user_id == user_id)

            query = query.order_by(self.model.updated_at.desc())

            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting projects by sample rate: {str(e)}")


# Import Track here to avoid circular imports
from ..models import Track