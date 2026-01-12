"""
Effect Repository
Specialized repository for Effect model operations
"""

import uuid
from typing import List, Optional, Dict, Any
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import Effect
from ..schemas import EffectCreate, EffectUpdate
from .base import BaseRepository, RepositoryError


class EffectRepository(BaseRepository[Effect, EffectCreate, EffectUpdate]):
    """Repository for Effect operations"""

    def __init__(self, session: AsyncSession):
        super().__init__(Effect, session)

    async def get_track_effects(
        self,
        track_id: uuid.UUID,
        active_only: bool = True
    ) -> List[Effect]:
        """Get all effects for a track, ordered by order_index"""
        try:
            query = (
                select(self.model)
                .where(self.model.track_id == track_id)
                .order_by(self.model.order_index)
            )

            if active_only:
                query = query.where(self.model.bypass == False)

            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting track effects: {str(e)}")

    async def get_clip_effects(
        self,
        clip_id: uuid.UUID,
        active_only: bool = True
    ) -> List[Effect]:
        """Get all effects for a clip, ordered by order_index"""
        try:
            query = (
                select(self.model)
                .where(self.model.clip_id == clip_id)
                .order_by(self.model.order_index)
            )

            if active_only:
                query = query.where(self.model.bypass == False)

            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting clip effects: {str(e)}")

    async def get_by_type(
        self,
        effect_type: str,
        track_id: Optional[uuid.UUID] = None,
        clip_id: Optional[uuid.UUID] = None
    ) -> List[Effect]:
        """Get effects by type, optionally filtered by track or clip"""
        try:
            query = select(self.model).where(self.model.effect_type == effect_type)

            if track_id:
                query = query.where(self.model.track_id == track_id)
            elif clip_id:
                query = query.where(self.model.clip_id == clip_id)

            query = query.order_by(self.model.order_index)

            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting effects by type: {str(e)}")

    async def get_next_order_index(
        self,
        track_id: Optional[uuid.UUID] = None,
        clip_id: Optional[uuid.UUID] = None
    ) -> int:
        """Get the next available order index for an effect chain"""
        try:
            query = select(func.coalesce(func.max(self.model.order_index), -1) + 1)

            if track_id:
                query = query.where(self.model.track_id == track_id)
            elif clip_id:
                query = query.where(self.model.clip_id == clip_id)
            else:
                raise RepositoryError("Either track_id or clip_id must be provided")

            result = await self.session.execute(query)
            return result.scalar()
        except Exception as e:
            raise RepositoryError(f"Error getting next order index: {str(e)}")

    async def reorder_effects(
        self,
        effect_ids: List[uuid.UUID],
        track_id: Optional[uuid.UUID] = None,
        clip_id: Optional[uuid.UUID] = None
    ) -> bool:
        """Reorder effects in a chain"""
        try:
            for index, effect_id in enumerate(effect_ids):
                effect = await self.get_or_404(effect_id)

                # Validate ownership
                if track_id and effect.track_id != track_id:
                    raise RepositoryError("Effect does not belong to track")
                elif clip_id and effect.clip_id != clip_id:
                    raise RepositoryError("Effect does not belong to clip")

                effect.order_index = index

            await self.session.flush()
            return True
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error reordering effects: {str(e)}")

    async def toggle_bypass(self, effect_id: uuid.UUID) -> Effect:
        """Toggle the bypass state of an effect"""
        try:
            effect = await self.get_or_404(effect_id)
            effect.bypass = not effect.bypass
            await self.session.flush()
            await self.session.refresh(effect)
            return effect
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error toggling effect bypass: {str(e)}")

    async def update_parameters(
        self,
        effect_id: uuid.UUID,
        parameters: Dict[str, Any]
    ) -> Effect:
        """Update effect parameters"""
        try:
            effect = await self.get_or_404(effect_id)

            # Merge new parameters with existing ones
            updated_params = {**effect.parameters, **parameters}
            effect.parameters = updated_params

            await self.session.flush()
            await self.session.refresh(effect)
            return effect
        except Exception as e:
            await self.session.rollback()
            raise RepositoryError(f"Error updating effect parameters: {str(e)}")

    async def get_project_effects(self, project_id: uuid.UUID) -> List[Effect]:
        """Get all effects in a project across all tracks and clips"""
        try:
            from ..models import Track, Clip

            # Get track effects
            track_effects_query = (
                select(self.model)
                .join(Track)
                .where(Track.project_id == project_id)
                .where(self.model.track_id.isnot(None))
            )

            # Get clip effects
            clip_effects_query = (
                select(self.model)
                .join(Clip)
                .join(Track, Clip.track_id == Track.id)
                .where(Track.project_id == project_id)
                .where(self.model.clip_id.isnot(None))
            )

            # Combine queries
            from sqlalchemy import union_all
            combined_query = union_all(track_effects_query, clip_effects_query)

            result = await self.session.execute(combined_query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting project effects: {str(e)}")

    async def count_by_type(self, effect_type: str) -> int:
        """Count effects by type across the system"""
        try:
            result = await self.session.execute(
                select(func.count(self.model.id))
                .where(self.model.effect_type == effect_type)
            )
            return result.scalar()
        except Exception as e:
            raise RepositoryError(f"Error counting effects by type: {str(e)}")

    async def get_active_effects(
        self,
        track_id: Optional[uuid.UUID] = None,
        clip_id: Optional[uuid.UUID] = None
    ) -> List[Effect]:
        """Get all active (non-bypassed) effects for track or clip"""
        try:
            query = (
                select(self.model)
                .where(self.model.bypass == False)
                .order_by(self.model.order_index)
            )

            if track_id:
                query = query.where(self.model.track_id == track_id)
            elif clip_id:
                query = query.where(self.model.clip_id == clip_id)
            else:
                raise RepositoryError("Either track_id or clip_id must be provided")

            result = await self.session.execute(query)
            return result.scalars().all()
        except Exception as e:
            raise RepositoryError(f"Error getting active effects: {str(e)}")