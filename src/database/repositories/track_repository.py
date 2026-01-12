"""
Track Repository
Specialized repository for Track model operations
"""

import uuid
from typing import List, Optional
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import Track
from ..schemas import TrackCreate, TrackUpdate
from .base import BaseRepository, RepositoryError


class TrackRepository(BaseRepository[Track, TrackCreate, TrackUpdate]):
    """Repository for Track operations"""

    def __init__(self, session: AsyncSession):
        super().__init__(Track, session)

    async def get_by_project(
        self,
        project_id: uuid.UUID,
        include_clips: bool = False
    ) -> List[Track]:
        """Get all tracks for a project, ordered by track_index"""
        try:
            query = (
                select(self.model)
                .where(self.model.project_id == project_id)
                .order_by(self.model.track_index)
            )

            if include_clips:
                query = query.options(
                    selectinload(self.model.clips),
                    selectinload(self.model.effects)
                )

            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting project tracks: {str(e)}")

    async def get_next_track_index(self, project_id: uuid.UUID) -> int:
        """Get the next available track index for a project"""
        try:
            result = await self.session.execute(
                select(func.coalesce(func.max(self.model.track_index), -1) + 1)
                .where(self.model.project_id == project_id)
            )
            return result.scalar()
        except Exception as e:
            raise RepositoryError(f"Error getting next track index: {str(e)}")

    async def reorder_tracks(
        self,
        project_id: uuid.UUID,
        track_order: List[uuid.UUID]
    ) -> bool:
        """Reorder tracks in a project"""
        try:
            for index, track_id in enumerate(track_order):
                track = await self.get_or_404(track_id)
                if track.project_id != project_id:
                    raise RepositoryError("Track does not belong to project")
                track.track_index = index

            await self.session.flush()
            return True
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error reordering tracks: {str(e)}")

    async def get_muted_tracks(self, project_id: uuid.UUID) -> List[Track]:
        """Get all muted tracks in a project"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(
                    (self.model.project_id == project_id) &
                    (self.model.muted == True)
                )
                .order_by(self.model.track_index)
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting muted tracks: {str(e)}")

    async def get_soloed_tracks(self, project_id: uuid.UUID) -> List[Track]:
        """Get all soloed tracks in a project"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(
                    (self.model.project_id == project_id) &
                    (self.model.soloed == True)
                )
                .order_by(self.model.track_index)
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting soloed tracks: {str(e)}")

    async def count_by_project(self, project_id: uuid.UUID) -> int:
        """Count tracks in a project"""
        try:
            result = await self.session.execute(
                select(func.count(self.model.id))
                .where(self.model.project_id == project_id)
            )
            return result.scalar()
        except Exception as e:
            raise RepositoryError(f"Error counting tracks: {str(e)}")

    async def bulk_mute(self, track_ids: List[uuid.UUID], muted: bool = True) -> int:
        """Bulk mute/unmute tracks"""
        try:
            from sqlalchemy import update
            stmt = (
                update(self.model)
                .where(self.model.id.in_(track_ids))
                .values(muted=muted)
            )
            result = await self.session.execute(stmt)
            await self.session.flush()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error bulk muting tracks: {str(e)}")

    async def bulk_solo(self, track_ids: List[uuid.UUID], soloed: bool = True) -> int:
        """Bulk solo/unsolo tracks"""
        try:
            from sqlalchemy import update
            stmt = (
                update(self.model)
                .where(self.model.id.in_(track_ids))
                .values(soloed=soloed)
            )
            result = await self.session.execute(stmt)
            await self.session.flush()
            return result.rowcount
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error bulk soloing tracks: {str(e)}")