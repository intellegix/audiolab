"""
AudioLab Loop Manager
Loop region management and automatic looping functionality
"""

import uuid
import asyncio
import logging
from typing import Optional, List, Dict, Any, Callable
from decimal import Decimal
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import LoopRegion, Project
from ..database.connection import get_session

logger = logging.getLogger(__name__)


@dataclass
class LoopRegionData:
    """Runtime loop region data"""
    id: uuid.UUID
    project_id: uuid.UUID
    name: str
    start_time: float
    end_time: float
    is_enabled: bool
    repeat_count: Optional[int]  # None = infinite
    auto_punch_record: bool
    color: Optional[str] = None
    notes: Optional[str] = None


class LoopManager:
    """
    Manages loop regions and automatic looping behavior during playback
    Handles loop boundary detection, restart logic, and punch recording
    """

    def __init__(self, playback_engine=None):
        """
        Initialize loop manager

        Args:
            playback_engine: PlaybackEngine instance for position control
        """
        self.playback_engine = playback_engine

        # Loop state
        self.active_loop: Optional[LoopRegionData] = None
        self.loop_count = 0
        self.last_position = 0.0

        # Project loop regions
        self.loaded_regions: Dict[uuid.UUID, LoopRegionData] = {}
        self.current_project_id: Optional[uuid.UUID] = None

        # Callbacks
        self.loop_start_callbacks: List[Callable[[LoopRegionData, int], None]] = []
        self.loop_end_callbacks: List[Callable[[LoopRegionData, int], None]] = []

        # Recording integration
        self.auto_record_callback: Optional[Callable[[LoopRegionData, bool], None]] = None

        logger.info("LoopManager initialized")

    def set_playback_engine(self, playback_engine):
        """Set playback engine reference"""
        self.playback_engine = playback_engine

    async def load_project_loops(self, project_id: uuid.UUID) -> int:
        """
        Load loop regions for a project

        Args:
            project_id: Project ID to load loops for

        Returns:
            Number of loop regions loaded
        """
        try:
            self.current_project_id = project_id
            self.loaded_regions.clear()
            self.active_loop = None

            async with get_session() as db:
                # Load all loop regions for project
                stmt = (
                    select(LoopRegion)
                    .where(LoopRegion.project_id == project_id)
                    .order_by(LoopRegion.start_time)
                )

                result = await db.execute(stmt)
                loop_regions = result.scalars().all()

                # Convert to runtime data
                for region in loop_regions:
                    loop_data = LoopRegionData(
                        id=region.id,
                        project_id=region.project_id,
                        name=region.name,
                        start_time=float(region.start_time),
                        end_time=float(region.end_time),
                        is_enabled=region.is_enabled,
                        repeat_count=region.repeat_count,
                        auto_punch_record=region.auto_punch_record,
                        color=region.color,
                        notes=region.notes
                    )
                    self.loaded_regions[region.id] = loop_data

                logger.info(f"Loaded {len(self.loaded_regions)} loop regions for project {project_id}")
                return len(self.loaded_regions)

        except Exception as e:
            logger.error(f"Error loading project loops: {e}")
            return 0

    async def enable_loop(self, loop_region_id: uuid.UUID) -> bool:
        """
        Enable looping for specified region

        Args:
            loop_region_id: Loop region ID to activate

        Returns:
            True if loop enabled successfully
        """
        try:
            if loop_region_id not in self.loaded_regions:
                logger.warning(f"Loop region {loop_region_id} not found in loaded regions")
                return False

            loop_region = self.loaded_regions[loop_region_id]

            if not loop_region.is_enabled:
                logger.warning(f"Loop region '{loop_region.name}' is disabled")
                return False

            # Validate loop region
            if loop_region.start_time >= loop_region.end_time:
                logger.error(f"Invalid loop region: start_time >= end_time")
                return False

            self.active_loop = loop_region
            self.loop_count = 0
            logger.info(f"Loop enabled: '{loop_region.name}' ({loop_region.start_time:.3f}s - {loop_region.end_time:.3f}s)")

            # Start auto punch recording if enabled
            if loop_region.auto_punch_record and self.auto_record_callback:
                await self.auto_record_callback(loop_region, True)

            return True

        except Exception as e:
            logger.error(f"Error enabling loop: {e}")
            return False

    async def disable_loop(self) -> bool:
        """
        Disable active loop

        Returns:
            True if loop disabled successfully
        """
        try:
            if not self.active_loop:
                return True

            # Stop auto punch recording if active
            if self.active_loop.auto_punch_record and self.auto_record_callback:
                await self.auto_record_callback(self.active_loop, False)

            logger.info(f"Loop disabled: '{self.active_loop.name}'")
            self.active_loop = None
            self.loop_count = 0

            return True

        except Exception as e:
            logger.error(f"Error disabling loop: {e}")
            return False

    async def check_loop_boundary(self, current_position: float) -> bool:
        """
        Check if playback has hit loop boundary and restart if needed

        Args:
            current_position: Current playback position in seconds

        Returns:
            True if loop restart occurred
        """
        if not self.active_loop or not self.playback_engine:
            self.last_position = current_position
            return False

        try:
            # Check if we've crossed the loop end boundary
            if (self.last_position < self.active_loop.end_time and
                current_position >= self.active_loop.end_time):

                # Check repeat count limit
                if (self.active_loop.repeat_count is not None and
                    self.loop_count >= self.active_loop.repeat_count):
                    logger.info(f"Loop '{self.active_loop.name}' completed {self.loop_count} repeats")
                    await self.disable_loop()
                    self.last_position = current_position
                    return False

                # Restart loop
                await self._restart_loop()
                return True

            # Check if playback jumped before loop start (seek operation)
            elif (self.last_position >= self.active_loop.start_time and
                  current_position < self.active_loop.start_time):
                # Reset loop count if we seek before loop start
                self.loop_count = 0

            self.last_position = current_position
            return False

        except Exception as e:
            logger.error(f"Error checking loop boundary: {e}")
            self.last_position = current_position
            return False

    async def _restart_loop(self):
        """Restart active loop from beginning"""
        try:
            if not self.active_loop or not self.playback_engine:
                return

            # Increment loop count
            self.loop_count += 1

            # Notify callbacks of loop end
            for callback in self.loop_end_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self.active_loop, self.loop_count)
                    else:
                        callback(self.active_loop, self.loop_count)
                except Exception as e:
                    logger.error(f"Error in loop end callback: {e}")

            # Seek to loop start
            await self.playback_engine.seek_to(self.active_loop.start_time)
            self.last_position = self.active_loop.start_time

            # Notify callbacks of loop start
            for callback in self.loop_start_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(self.active_loop, self.loop_count)
                    else:
                        callback(self.active_loop, self.loop_count)
                except Exception as e:
                    logger.error(f"Error in loop start callback: {e}")

            logger.debug(f"Loop restarted: '{self.active_loop.name}' (iteration {self.loop_count})")

        except Exception as e:
            logger.error(f"Error restarting loop: {e}")

    def get_active_loop(self) -> Optional[LoopRegionData]:
        """Get currently active loop region"""
        return self.active_loop

    def get_loaded_regions(self) -> List[LoopRegionData]:
        """Get all loaded loop regions"""
        return list(self.loaded_regions.values())

    def get_loop_status(self) -> Dict[str, Any]:
        """Get current loop status"""
        status = {
            "active": self.active_loop is not None,
            "loop_count": self.loop_count,
            "loaded_regions": len(self.loaded_regions),
            "project_id": str(self.current_project_id) if self.current_project_id else None
        }

        if self.active_loop:
            status.update({
                "loop_id": str(self.active_loop.id),
                "loop_name": self.active_loop.name,
                "start_time": self.active_loop.start_time,
                "end_time": self.active_loop.end_time,
                "duration": self.active_loop.end_time - self.active_loop.start_time,
                "repeat_count": self.active_loop.repeat_count,
                "auto_punch_record": self.active_loop.auto_punch_record
            })

        return status

    def add_loop_start_callback(self, callback: Callable[[LoopRegionData, int], None]):
        """
        Add callback for loop start events

        Args:
            callback: Function called when loop restarts (loop_region, iteration)
        """
        if callback not in self.loop_start_callbacks:
            self.loop_start_callbacks.append(callback)

    def add_loop_end_callback(self, callback: Callable[[LoopRegionData, int], None]):
        """
        Add callback for loop end events

        Args:
            callback: Function called when loop ends (loop_region, iteration)
        """
        if callback not in self.loop_end_callbacks:
            self.loop_end_callbacks.append(callback)

    def set_auto_record_callback(self, callback: Callable[[LoopRegionData, bool], None]):
        """
        Set callback for auto punch recording

        Args:
            callback: Function called for punch in/out (loop_region, punch_in)
        """
        self.auto_record_callback = callback

    async def create_loop_region(self,
                                project_id: uuid.UUID,
                                name: str,
                                start_time: float,
                                end_time: float,
                                repeat_count: Optional[int] = None,
                                auto_punch_record: bool = False) -> Optional[uuid.UUID]:
        """
        Create new loop region

        Args:
            project_id: Project ID
            name: Loop region name
            start_time: Start position in seconds
            end_time: End position in seconds
            repeat_count: Number of repeats (None = infinite)
            auto_punch_record: Enable auto punch recording

        Returns:
            Loop region ID if created successfully
        """
        try:
            if start_time >= end_time:
                logger.error("Invalid loop region: start_time must be less than end_time")
                return None

            async with get_session() as db:
                loop_region = LoopRegion(
                    project_id=project_id,
                    name=name,
                    start_time=Decimal(str(start_time)),
                    end_time=Decimal(str(end_time)),
                    repeat_count=repeat_count,
                    auto_punch_record=auto_punch_record
                )

                db.add(loop_region)
                await db.commit()
                await db.refresh(loop_region)

                # Add to loaded regions if it's for current project
                if project_id == self.current_project_id:
                    loop_data = LoopRegionData(
                        id=loop_region.id,
                        project_id=loop_region.project_id,
                        name=loop_region.name,
                        start_time=float(loop_region.start_time),
                        end_time=float(loop_region.end_time),
                        is_enabled=loop_region.is_enabled,
                        repeat_count=loop_region.repeat_count,
                        auto_punch_record=loop_region.auto_punch_record,
                        color=loop_region.color,
                        notes=loop_region.notes
                    )
                    self.loaded_regions[loop_region.id] = loop_data

                logger.info(f"Created loop region '{name}' ({start_time:.3f}s - {end_time:.3f}s)")
                return loop_region.id

        except Exception as e:
            logger.error(f"Error creating loop region: {e}")
            return None

    async def update_loop_region(self,
                                loop_id: uuid.UUID,
                                **updates) -> bool:
        """
        Update loop region properties

        Args:
            loop_id: Loop region ID
            **updates: Properties to update

        Returns:
            True if updated successfully
        """
        try:
            async with get_session() as db:
                stmt = select(LoopRegion).where(LoopRegion.id == loop_id)
                result = await db.execute(stmt)
                loop_region = result.scalar_one_or_none()

                if not loop_region:
                    logger.warning(f"Loop region {loop_id} not found")
                    return False

                # Update properties
                for key, value in updates.items():
                    if hasattr(loop_region, key):
                        if key in ['start_time', 'end_time'] and isinstance(value, (int, float)):
                            value = Decimal(str(value))
                        setattr(loop_region, key, value)

                await db.commit()

                # Update loaded regions cache
                if loop_id in self.loaded_regions:
                    await self.load_project_loops(self.current_project_id)

                # Disable active loop if it was modified
                if self.active_loop and self.active_loop.id == loop_id:
                    await self.disable_loop()

                logger.info(f"Updated loop region {loop_id}")
                return True

        except Exception as e:
            logger.error(f"Error updating loop region: {e}")
            return False

    async def delete_loop_region(self, loop_id: uuid.UUID) -> bool:
        """
        Delete loop region

        Args:
            loop_id: Loop region ID to delete

        Returns:
            True if deleted successfully
        """
        try:
            # Disable loop if it's currently active
            if self.active_loop and self.active_loop.id == loop_id:
                await self.disable_loop()

            async with get_session() as db:
                stmt = select(LoopRegion).where(LoopRegion.id == loop_id)
                result = await db.execute(stmt)
                loop_region = result.scalar_one_or_none()

                if loop_region:
                    await db.delete(loop_region)
                    await db.commit()

                    # Remove from loaded regions
                    if loop_id in self.loaded_regions:
                        del self.loaded_regions[loop_id]

                    logger.info(f"Deleted loop region {loop_id}")
                    return True
                else:
                    logger.warning(f"Loop region {loop_id} not found")
                    return False

        except Exception as e:
            logger.error(f"Error deleting loop region: {e}")
            return False

    def cleanup(self):
        """Cleanup loop manager resources"""
        try:
            self.active_loop = None
            self.loaded_regions.clear()
            self.loop_start_callbacks.clear()
            self.loop_end_callbacks.clear()
            self.auto_record_callback = None

            logger.debug("LoopManager cleanup completed")

        except Exception as e:
            logger.error(f"Error during LoopManager cleanup: {e}")


# Global instance for application use
_loop_manager: Optional[LoopManager] = None


def get_loop_manager() -> LoopManager:
    """
    Get singleton LoopManager instance

    Returns:
        Global LoopManager instance
    """
    global _loop_manager
    if _loop_manager is None:
        _loop_manager = LoopManager()
    return _loop_manager


async def cleanup_loop_manager():
    """Cleanup loop manager resources"""
    global _loop_manager
    if _loop_manager is not None:
        _loop_manager.cleanup()
        _loop_manager = None
    logger.info("Loop manager cleanup completed")