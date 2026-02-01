"""
Beat Generation Repository
Database operations for beat generation requests, templates, and variations
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, and_, or_
from sqlalchemy.orm import selectinload

from ..models import BeatGenerationRequest, BeatTemplate, BeatVariation, Project
from ..schemas import (
    BeatGenerationRequest as BeatGenerationRequestSchema,
    BeatGenerationUpdate,
    BeatTemplateCreate,
    BeatTemplateUpdate,
    BeatVariationCreate,
    BeatVariationUpdate
)
from ...core.result import Result


class BeatGenerationRepository:
    """Repository for beat generation database operations"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_beat_generation_request(
        self,
        request_data: BeatGenerationRequestSchema,
        user_id: uuid.UUID
    ) -> Result[BeatGenerationRequest]:
        """Create a new beat generation request with project tempo sync"""
        try:
            # Get project tempo and time signature for synchronization
            project_query = select(Project).where(Project.id == request_data.project_id)
            project_result = await self.session.execute(project_query)
            project = project_result.scalar_one_or_none()

            if not project:
                return Result.err(f"Project {request_data.project_id} not found")

            # Create beat generation request
            beat_request = BeatGenerationRequest(
                project_id=request_data.project_id,
                user_id=user_id,
                prompt=request_data.prompt,
                provider=request_data.provider,
                model_name=request_data.model_name,
                duration=request_data.duration,
                # Sync with project settings
                tempo=project.tempo,
                time_signature=project.time_signature,
                style_tags=request_data.style_tags,
                status="pending",
                progress=Decimal("0.0")
            )

            self.session.add(beat_request)
            await self.session.commit()
            await self.session.refresh(beat_request)

            return Result.ok(beat_request)

        except Exception as e:
            await self.session.rollback()
            return Result.err(f"Failed to create beat generation request: {str(e)}")

    async def get_beat_generation_request(
        self,
        request_id: uuid.UUID
    ) -> Result[Optional[BeatGenerationRequest]]:
        """Get beat generation request by ID"""
        try:
            query = select(BeatGenerationRequest).where(BeatGenerationRequest.id == request_id)
            result = await self.session.execute(query)
            beat_request = result.scalar_one_or_none()
            return Result.ok(beat_request)

        except Exception as e:
            return Result.err(f"Failed to get beat generation request: {str(e)}")

    async def update_beat_generation_request(
        self,
        request_id: uuid.UUID,
        update_data: BeatGenerationUpdate
    ) -> Result[BeatGenerationRequest]:
        """Update beat generation request status and results"""
        try:
            # Build update dictionary with non-None values
            update_dict = {}
            for field, value in update_data.model_dump(exclude_unset=True).items():
                if value is not None:
                    update_dict[field] = value

            # Add timestamp updates based on status
            if update_data.status == "processing" and "started_at" not in update_dict:
                update_dict["started_at"] = datetime.utcnow()
            elif update_data.status in ["completed", "failed"] and "completed_at" not in update_dict:
                update_dict["completed_at"] = datetime.utcnow()

            # Execute update
            query = (
                update(BeatGenerationRequest)
                .where(BeatGenerationRequest.id == request_id)
                .values(**update_dict)
                .returning(BeatGenerationRequest)
            )
            result = await self.session.execute(query)
            updated_request = result.scalar_one_or_none()

            if not updated_request:
                return Result.err(f"Beat generation request {request_id} not found")

            await self.session.commit()
            return Result.ok(updated_request)

        except Exception as e:
            await self.session.rollback()
            return Result.err(f"Failed to update beat generation request: {str(e)}")

    async def get_user_beat_requests(
        self,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0
    ) -> Result[List[BeatGenerationRequest]]:
        """Get beat generation requests for a user"""
        try:
            query = (
                select(BeatGenerationRequest)
                .where(BeatGenerationRequest.user_id == user_id)
                .order_by(BeatGenerationRequest.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self.session.execute(query)
            requests = result.scalars().all()
            return Result.ok(list(requests))

        except Exception as e:
            return Result.err(f"Failed to get user beat requests: {str(e)}")

    async def get_project_beat_requests(
        self,
        project_id: uuid.UUID
    ) -> Result[List[BeatGenerationRequest]]:
        """Get all beat generation requests for a project"""
        try:
            query = (
                select(BeatGenerationRequest)
                .where(BeatGenerationRequest.project_id == project_id)
                .order_by(BeatGenerationRequest.created_at.desc())
            )
            result = await self.session.execute(query)
            requests = result.scalars().all()
            return Result.ok(list(requests))

        except Exception as e:
            return Result.err(f"Failed to get project beat requests: {str(e)}")

    # Beat Template Operations
    async def create_beat_template(
        self,
        template_data: BeatTemplateCreate
    ) -> Result[BeatTemplate]:
        """Create a new beat template"""
        try:
            template = BeatTemplate(
                name=template_data.name,
                description=template_data.description,
                category=template_data.category,
                tags=template_data.tags,
                default_tempo=template_data.default_tempo,
                time_signature=template_data.time_signature,
                duration=template_data.duration,
                provider_config=template_data.provider_config,
                prompt_template=template_data.prompt_template,
                is_public=template_data.is_public,
                created_by_user_id=template_data.created_by_user_id
            )

            self.session.add(template)
            await self.session.commit()
            await self.session.refresh(template)

            return Result.ok(template)

        except Exception as e:
            await self.session.rollback()
            return Result.err(f"Failed to create beat template: {str(e)}")

    async def get_beat_templates(
        self,
        category: Optional[str] = None,
        is_public: Optional[bool] = None,
        search_tags: Optional[List[str]] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Result[List[BeatTemplate]]:
        """Get beat templates with optional filtering"""
        try:
            query = select(BeatTemplate)

            # Apply filters
            if category:
                query = query.where(BeatTemplate.category == category)

            if is_public is not None:
                query = query.where(BeatTemplate.is_public == is_public)

            if search_tags:
                # Search for templates containing any of the search tags
                tag_conditions = [
                    func.jsonb_exists(BeatTemplate.tags, tag) for tag in search_tags
                ]
                query = query.where(or_(*tag_conditions))

            query = (
                query
                .order_by(BeatTemplate.usage_count.desc(), BeatTemplate.created_at.desc())
                .limit(limit)
                .offset(offset)
            )

            result = await self.session.execute(query)
            templates = result.scalars().all()
            return Result.ok(list(templates))

        except Exception as e:
            return Result.err(f"Failed to get beat templates: {str(e)}")

    async def update_template_usage(
        self,
        template_id: uuid.UUID,
        quality_score: Optional[Decimal] = None
    ) -> Result[BeatTemplate]:
        """Update template usage count and average quality"""
        try:
            # Get current template
            template_query = select(BeatTemplate).where(BeatTemplate.id == template_id)
            template_result = await self.session.execute(template_query)
            template = template_result.scalar_one_or_none()

            if not template:
                return Result.err(f"Beat template {template_id} not found")

            # Update usage count
            new_usage_count = template.usage_count + 1

            # Calculate new average quality if provided
            new_average_quality = template.average_quality
            if quality_score is not None:
                if template.average_quality is not None:
                    # Weighted average: (old_avg * old_count + new_score) / new_count
                    total_quality = template.average_quality * template.usage_count + quality_score
                    new_average_quality = total_quality / new_usage_count
                else:
                    new_average_quality = quality_score

            # Update template
            update_query = (
                update(BeatTemplate)
                .where(BeatTemplate.id == template_id)
                .values(
                    usage_count=new_usage_count,
                    average_quality=new_average_quality,
                    updated_at=datetime.utcnow()
                )
                .returning(BeatTemplate)
            )
            result = await self.session.execute(update_query)
            updated_template = result.scalar_one()

            await self.session.commit()
            return Result.ok(updated_template)

        except Exception as e:
            await self.session.rollback()
            return Result.err(f"Failed to update template usage: {str(e)}")

    # Beat Variation Operations
    async def create_beat_variation(
        self,
        variation_data: BeatVariationCreate
    ) -> Result[BeatVariation]:
        """Create a new beat variation"""
        try:
            variation = BeatVariation(
                beat_generation_request_id=variation_data.beat_generation_request_id,
                variation_index=variation_data.variation_index,
                name=variation_data.name,
                audio_path=variation_data.audio_path,
                midi_path=variation_data.midi_path,
                quality_score=variation_data.quality_score,
                user_rating=variation_data.user_rating,
                generation_seed=variation_data.generation_seed,
                generation_metadata=variation_data.generation_metadata,
                is_selected=variation_data.is_selected,
                used_in_project=variation_data.used_in_project
            )

            self.session.add(variation)
            await self.session.commit()
            await self.session.refresh(variation)

            return Result.ok(variation)

        except Exception as e:
            await self.session.rollback()
            return Result.err(f"Failed to create beat variation: {str(e)}")

    async def get_request_variations(
        self,
        request_id: uuid.UUID
    ) -> Result[List[BeatVariation]]:
        """Get all variations for a beat generation request"""
        try:
            query = (
                select(BeatVariation)
                .where(BeatVariation.beat_generation_request_id == request_id)
                .order_by(BeatVariation.variation_index)
            )
            result = await self.session.execute(query)
            variations = result.scalars().all()
            return Result.ok(list(variations))

        except Exception as e:
            return Result.err(f"Failed to get request variations: {str(e)}")

    async def select_variation(
        self,
        variation_id: uuid.UUID
    ) -> Result[BeatVariation]:
        """Mark a variation as selected and unselect others in the same request"""
        try:
            # Get the variation to select
            variation_query = select(BeatVariation).where(BeatVariation.id == variation_id)
            variation_result = await self.session.execute(variation_query)
            variation = variation_result.scalar_one_or_none()

            if not variation:
                return Result.err(f"Beat variation {variation_id} not found")

            # Unselect all other variations in the same request
            unselect_query = (
                update(BeatVariation)
                .where(
                    and_(
                        BeatVariation.beat_generation_request_id == variation.beat_generation_request_id,
                        BeatVariation.id != variation_id
                    )
                )
                .values(is_selected=False)
            )
            await self.session.execute(unselect_query)

            # Select the target variation
            select_query = (
                update(BeatVariation)
                .where(BeatVariation.id == variation_id)
                .values(is_selected=True)
                .returning(BeatVariation)
            )
            result = await self.session.execute(select_query)
            selected_variation = result.scalar_one()

            await self.session.commit()
            return Result.ok(selected_variation)

        except Exception as e:
            await self.session.rollback()
            return Result.err(f"Failed to select variation: {str(e)}")

    async def mark_variation_used(
        self,
        variation_id: uuid.UUID
    ) -> Result[BeatVariation]:
        """Mark a variation as used in project"""
        try:
            query = (
                update(BeatVariation)
                .where(BeatVariation.id == variation_id)
                .values(used_in_project=True)
                .returning(BeatVariation)
            )
            result = await self.session.execute(query)
            variation = result.scalar_one_or_none()

            if not variation:
                return Result.err(f"Beat variation {variation_id} not found")

            await self.session.commit()
            return Result.ok(variation)

        except Exception as e:
            await self.session.rollback()
            return Result.err(f"Failed to mark variation as used: {str(e)}")