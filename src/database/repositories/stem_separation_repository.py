"""
Stem Separation Repository
Specialized repository for StemSeparation model operations
"""

import uuid
from decimal import Decimal
from typing import List, Optional, Dict
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import StemSeparation
from ..schemas import StemSeparationCreate, StemSeparationUpdate
from .base import BaseRepository, RepositoryError


class StemSeparationRepository(BaseRepository[StemSeparation, StemSeparationCreate, StemSeparationUpdate]):
    """Repository for StemSeparation operations"""

    def __init__(self, session: AsyncSession):
        super().__init__(StemSeparation, session)

    async def get_by_clip(self, clip_id: uuid.UUID) -> List[StemSeparation]:
        """Get all stem separations for a clip, ordered by creation time"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(self.model.clip_id == clip_id)
                .order_by(desc(self.model.created_at))
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting clip separations: {str(e)}")

    async def get_latest_by_clip(self, clip_id: uuid.UUID) -> Optional[StemSeparation]:
        """Get the most recent stem separation for a clip"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(self.model.clip_id == clip_id)
                .order_by(desc(self.model.created_at))
                .limit(1)
            )
            return result.scalar_one_or_none()
        except Exception as e:
            raise RepositoryError(f"Error getting latest separation: {str(e)}")

    async def get_by_model(self, model_name: str) -> List[StemSeparation]:
        """Get all separations created with a specific model"""
        try:
            result = await self.session.execute(
                select(self.model)
                .where(self.model.model_used == model_name)
                .order_by(desc(self.model.created_at))
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting separations by model: {str(e)}")

    async def get_by_quality_threshold(
        self,
        min_quality: Decimal,
        model_name: Optional[str] = None
    ) -> List[StemSeparation]:
        """Get separations above a quality threshold"""
        try:
            query = (
                select(self.model)
                .where(
                    (self.model.quality_score.isnot(None)) &
                    (self.model.quality_score >= min_quality)
                )
                .order_by(desc(self.model.quality_score))
            )

            if model_name:
                query = query.where(self.model.model_used == model_name)

            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting separations by quality: {str(e)}")

    async def get_project_separations(self, project_id: uuid.UUID) -> List[StemSeparation]:
        """Get all stem separations in a project"""
        try:
            from ..models import Clip, Track

            result = await self.session.execute(
                select(self.model)
                .join(Clip)
                .join(Track)
                .where(Track.project_id == project_id)
                .order_by(desc(self.model.created_at))
            )
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting project separations: {str(e)}")

    async def get_processing_stats(
        self,
        model_name: Optional[str] = None
    ) -> Dict[str, float]:
        """Get processing statistics for stem separations"""
        try:
            query = select(
                func.avg(self.model.processing_time).label('avg_time'),
                func.min(self.model.processing_time).label('min_time'),
                func.max(self.model.processing_time).label('max_time'),
                func.avg(self.model.quality_score).label('avg_quality'),
                func.count(self.model.id).label('total_count')
            ).where(
                (self.model.processing_time.isnot(None)) &
                (self.model.quality_score.isnot(None))
            )

            if model_name:
                query = query.where(self.model.model_used == model_name)

            result = await self.session.execute(query)
            row = result.one()

            return {
                'average_processing_time': float(row.avg_time or 0),
                'min_processing_time': float(row.min_time or 0),
                'max_processing_time': float(row.max_time or 0),
                'average_quality_score': float(row.avg_quality or 0),
                'total_separations': int(row.total_count or 0)
            }
        except Exception as e:
            raise RepositoryError(f"Error getting processing stats: {str(e)}")

    async def get_stem_files(self, separation_id: uuid.UUID) -> Dict[str, str]:
        """Get stem file paths for a separation"""
        try:
            separation = await self.get_or_404(separation_id)
            return separation.stems
        except Exception as e:
            raise RepositoryError(f"Error getting stem files: {str(e)}")

    async def update_quality_score(
        self,
        separation_id: uuid.UUID,
        quality_score: Decimal
    ) -> StemSeparation:
        """Update the quality score for a separation"""
        try:
            separation = await self.get_or_404(separation_id)
            separation.quality_score = quality_score
            await self.session.flush()
            await self.session.refresh(separation)
            return separation
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error updating quality score: {str(e)}")

    async def count_by_model(self) -> Dict[str, int]:
        """Get separation count by model"""
        try:
            result = await self.session.execute(
                select(
                    self.model.model_used,
                    func.count(self.model.id).label('count')
                )
                .group_by(self.model.model_used)
                .order_by(desc(func.count(self.model.id)))
            )

            return {row.model_used: row.count for row in result}
        except Exception as e:
            raise RepositoryError(f"Error counting by model: {str(e)}")

    async def get_recent_separations(
        self,
        limit: int = 10,
        user_id: Optional[uuid.UUID] = None
    ) -> List[StemSeparation]:
        """Get recent stem separations, optionally filtered by user"""
        try:
            from ..models import Clip, Track, Project

            query = (
                select(self.model)
                .join(Clip)
                .join(Track)
                .join(Project)
                .order_by(desc(self.model.created_at))
                .limit(limit)
            )

            if user_id:
                query = query.where(Project.user_id == user_id)

            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting recent separations: {str(e)}")

    async def cleanup_orphaned_files(self) -> List[str]:
        """Get stem file paths for separations that no longer exist"""
        try:
            # This would be used by a cleanup service to remove orphaned audio files
            # Returns list of file paths that can be safely deleted
            result = await self.session.execute(
                select(self.model.stems)
                .where(
                    # Add conditions for orphaned separations
                    # For example, separations older than X days with no recent access
                    self.model.created_at < func.now() - func.interval('30 days')
                )
            )

            orphaned_files = []
            for row in result:
                stems = row.stems
                if isinstance(stems, dict):
                    orphaned_files.extend(stems.values())

            return orphaned_files
        except Exception as e:
            raise RepositoryError(f"Error finding orphaned files: {str(e)}")