"""
Clip Repository
Specialized repository for Clip model operations
"""

import uuid
from decimal import Decimal
from typing import List, Optional
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..models import Clip
from ..schemas import ClipCreate, ClipUpdate
from .base import BaseRepository, RepositoryError


class ClipRepository(BaseRepository[Clip, ClipCreate, ClipUpdate]):
    """Repository for Clip operations"""

    def __init__(self, session: AsyncSession):
        super().__init__(Clip, session)

    async def get_by_track(
        self,
        track_id: uuid.UUID,
        include_effects: bool = False
    ) -> List[Clip]:
        """Get all clips for a track, ordered by start_time"""
        try:
            query = (
                select(self.model)
                .where(self.model.track_id == track_id)
                .order_by(self.model.start_time)
            )

            if include_effects:
                query = query.options(selectinload(self.model.effects))

            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting track clips: {str(e)}")

    async def get_by_time_range(
        self,
        track_id: uuid.UUID,
        start_time: Decimal,
        end_time: Decimal
    ) -> List[Clip]:
        """Get clips that overlap with a time range"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(
                    and_(
                        self.model.track_id == track_id,
                        self.model.start_time < end_time,
                        (self.model.start_time + self.model.duration) > start_time
                    )
                )
                .order_by(self.model.start_time)
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting clips by time range: {str(e)}")

    async def check_overlap(
        self,
        track_id: uuid.UUID,
        start_time: Decimal,
        duration: Decimal,
        exclude_clip_id: Optional[uuid.UUID] = None
    ) -> bool:
        """Check if a clip would overlap with existing clips"""
        try:
            end_time = start_time + duration
            query = (
                select(func.count(self.model.id))
                .where(
                    and_(
                        self.model.track_id == track_id,
                        self.model.start_time < end_time,
                        (self.model.start_time + self.model.duration) > start_time
                    )
                )
            )

            if exclude_clip_id:
                query = query.where(self.model.id != exclude_clip_id)

            result = await self.session.execute(query)
            return result.scalar() > 0
        except Exception as e:
            raise RepositoryError(f"Error checking clip overlap: {str(e)}")

    async def get_project_clips(self, project_id: uuid.UUID) -> List[Clip]:
        """Get all clips in a project across all tracks"""
        try:
            from ..models import Track
            result = await self.session.execute(
                select(self.model)
                .join(Track)
                .where(Track.project_id == project_id)
                .order_by(Track.track_index, self.model.start_time)
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting project clips: {str(e)}")

    async def get_by_file_path(self, file_path: str) -> List[Clip]:
        """Get clips that reference a specific audio file"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(self.model.file_path == file_path)
                .order_by(self.model.created_at)
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting clips by file path: {str(e)}")

    async def get_timeline_duration(self, track_id: uuid.UUID) -> Optional[Decimal]:
        """Get the total timeline duration for a track (end of last clip)"""
        try:
            result = await self.session.execute(
                select(func.max(self.model.start_time + self.model.duration))
                .where(self.model.track_id == track_id)
            )
            return result.scalar()
        except Exception as e:
            raise RepositoryError(f"Error getting timeline duration: {str(e)}")

    async def move_clip(
        self,
        clip_id: uuid.UUID,
        new_start_time: Decimal,
        check_overlap: bool = True
    ) -> Clip:
        """Move a clip to a new timeline position"""
        try:
            clip = await self.get_or_404(clip_id)

            if check_overlap:
                has_overlap = await self.check_overlap(
                    clip.track_id,
                    new_start_time,
                    clip.duration,
                    exclude_clip_id=clip_id
                )
                if has_overlap:
                    raise RepositoryError("Clip would overlap with existing clips")

            clip.start_time = new_start_time
            await self.session.flush()
            await self.session.refresh(clip)
            return clip

        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error moving clip: {str(e)}")

    async def trim_clip(
        self,
        clip_id: uuid.UUID,
        new_start_time: Optional[Decimal] = None,
        new_duration: Optional[Decimal] = None
    ) -> Clip:
        """Trim a clip by adjusting start time and/or duration"""
        try:
            clip = await self.get_or_404(clip_id)

            if new_start_time is not None:
                # Adjust offset to maintain audio sync
                time_diff = new_start_time - clip.start_time
                clip.offset += time_diff
                clip.start_time = new_start_time
                clip.duration -= time_diff

            if new_duration is not None:
                clip.duration = new_duration

            # Ensure duration is positive
            if clip.duration <= 0:
                raise RepositoryError("Clip duration must be positive")

            await self.session.flush()
            await self.session.refresh(clip)
            return clip

        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error trimming clip: {str(e)}")