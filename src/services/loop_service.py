"""
AudioLab Loop Service
High-level loop management service with API integration
"""

import uuid
import logging
from typing import List, Optional, Dict, Any, Callable

from ..core.loop_manager import LoopManager, get_loop_manager, LoopRegionData
from ..services.playback_service import get_playback_service
from ..core.playback_engine import get_playback_engine

logger = logging.getLogger(__name__)


class LoopService:
    """
    High-level loop service for orchestrating loop functionality
    Integrates loop manager with playback engine and provides API-level functionality
    """

    def __init__(self):
        """Initialize loop service"""
        self._loop_manager: Optional[LoopManager] = None

        # WebSocket callback for real-time updates
        self._websocket_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

        logger.info("LoopService initialized")

    async def initialize(self):
        """Initialize loop service dependencies"""
        try:
            self._loop_manager = get_loop_manager()

            # Integrate with playback engine
            playback_engine = get_playback_engine()
            playback_engine.set_loop_manager(self._loop_manager)
            self._loop_manager.set_playback_engine(playback_engine)

            # Set up callbacks for WebSocket updates
            self._loop_manager.add_loop_start_callback(self._on_loop_start)
            self._loop_manager.add_loop_end_callback(self._on_loop_end)

            logger.info("LoopService dependencies initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LoopService: {e}")
            raise

    def set_websocket_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Set WebSocket callback for real-time loop updates

        Args:
            callback: Function to send WebSocket messages (connection_id, message)
        """
        self._websocket_callback = callback

    async def load_project_loops(self, project_id: uuid.UUID) -> Dict[str, Any]:
        """
        Load all loop regions for a project

        Args:
            project_id: Project ID to load loops for

        Returns:
            Load operation result with loop regions
        """
        try:
            if not self._loop_manager:
                await self.initialize()

            count = await self._loop_manager.load_project_loops(project_id)
            regions = self._loop_manager.get_loaded_regions()

            return {
                "success": True,
                "project_id": str(project_id),
                "regions_loaded": count,
                "regions": [self._loop_region_to_dict(region) for region in regions]
            }

        except Exception as e:
            logger.error(f"Error loading project loops: {e}")
            return {
                "success": False,
                "errors": [f"Failed to load project loops: {str(e)}"]
            }

    async def create_loop_region(self,
                                project_id: uuid.UUID,
                                name: str,
                                start_time: float,
                                end_time: float,
                                repeat_count: Optional[int] = None,
                                auto_punch_record: bool = False,
                                connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create new loop region

        Args:
            project_id: Project ID
            name: Loop region name
            start_time: Start position in seconds
            end_time: End position in seconds
            repeat_count: Number of repeats (None = infinite)
            auto_punch_record: Enable auto punch recording
            connection_id: WebSocket connection ID for updates

        Returns:
            Create operation result
        """
        try:
            if not self._loop_manager:
                await self.initialize()

            # Validate parameters
            if start_time >= end_time:
                return {
                    "success": False,
                    "errors": ["Start time must be less than end time"]
                }

            if end_time - start_time < 0.1:  # Minimum 100ms loop
                return {
                    "success": False,
                    "errors": ["Loop duration must be at least 0.1 seconds"]
                }

            # Create loop region
            loop_id = await self._loop_manager.create_loop_region(
                project_id=project_id,
                name=name,
                start_time=start_time,
                end_time=end_time,
                repeat_count=repeat_count,
                auto_punch_record=auto_punch_record
            )

            if loop_id:
                logger.info(f"Created loop region '{name}' ({start_time:.3f}s - {end_time:.3f}s)")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "loop_region_created",
                        "data": {
                            "loop_id": str(loop_id),
                            "project_id": str(project_id),
                            "name": name,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": end_time - start_time
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "loop_id": str(loop_id),
                    "name": name,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to create loop region"]
                }

        except Exception as e:
            logger.error(f"Error creating loop region: {e}")
            return {
                "success": False,
                "errors": [f"Create loop error: {str(e)}"]
            }

    async def enable_loop(self, loop_id: uuid.UUID, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Enable loop region for automatic looping

        Args:
            loop_id: Loop region ID to enable
            connection_id: WebSocket connection ID for updates

        Returns:
            Enable operation result
        """
        try:
            if not self._loop_manager:
                await self.initialize()

            success = await self._loop_manager.enable_loop(loop_id)

            if success:
                loop_region = self._loop_manager.get_active_loop()
                logger.info(f"Loop enabled: '{loop_region.name}' ({loop_region.start_time:.3f}s - {loop_region.end_time:.3f}s)")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "loop_enabled",
                        "data": {
                            "loop_id": str(loop_id),
                            "loop_region": self._loop_region_to_dict(loop_region)
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "loop_id": str(loop_id),
                    "loop_region": self._loop_region_to_dict(loop_region)
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to enable loop region"]
                }

        except Exception as e:
            logger.error(f"Error enabling loop: {e}")
            return {
                "success": False,
                "errors": [f"Enable loop error: {str(e)}"]
            }

    async def disable_loop(self, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Disable active loop

        Args:
            connection_id: WebSocket connection ID for updates

        Returns:
            Disable operation result
        """
        try:
            if not self._loop_manager:
                await self.initialize()

            active_loop = self._loop_manager.get_active_loop()
            success = await self._loop_manager.disable_loop()

            if success:
                logger.info("Loop disabled")

                # Send WebSocket notification
                if self._websocket_callback and connection_id and active_loop:
                    message = {
                        "type": "loop_disabled",
                        "data": {
                            "loop_id": str(active_loop.id)
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "message": "Loop disabled"
                }
            else:
                return {
                    "success": False,
                    "errors": ["No active loop to disable"]
                }

        except Exception as e:
            logger.error(f"Error disabling loop: {e}")
            return {
                "success": False,
                "errors": [f"Disable loop error: {str(e)}"]
            }

    async def update_loop_region(self, loop_id: uuid.UUID, **updates) -> Dict[str, Any]:
        """
        Update loop region properties

        Args:
            loop_id: Loop region ID
            **updates: Properties to update

        Returns:
            Update operation result
        """
        try:
            if not self._loop_manager:
                await self.initialize()

            # Validate time updates
            if 'start_time' in updates or 'end_time' in updates:
                current_region = None
                for region in self._loop_manager.get_loaded_regions():
                    if region.id == loop_id:
                        current_region = region
                        break

                if current_region:
                    start_time = updates.get('start_time', current_region.start_time)
                    end_time = updates.get('end_time', current_region.end_time)

                    if start_time >= end_time:
                        return {
                            "success": False,
                            "errors": ["Start time must be less than end time"]
                        }

                    if end_time - start_time < 0.1:
                        return {
                            "success": False,
                            "errors": ["Loop duration must be at least 0.1 seconds"]
                        }

            success = await self._loop_manager.update_loop_region(loop_id, **updates)

            if success:
                logger.info(f"Updated loop region {loop_id}")
                return {
                    "success": True,
                    "loop_id": str(loop_id),
                    "updated_properties": list(updates.keys())
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to update loop region"]
                }

        except Exception as e:
            logger.error(f"Error updating loop region: {e}")
            return {
                "success": False,
                "errors": [f"Update loop error: {str(e)}"]
            }

    async def delete_loop_region(self, loop_id: uuid.UUID, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete loop region

        Args:
            loop_id: Loop region ID to delete
            connection_id: WebSocket connection ID for updates

        Returns:
            Delete operation result
        """
        try:
            if not self._loop_manager:
                await self.initialize()

            success = await self._loop_manager.delete_loop_region(loop_id)

            if success:
                logger.info(f"Deleted loop region {loop_id}")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "loop_region_deleted",
                        "data": {
                            "loop_id": str(loop_id)
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "loop_id": str(loop_id)
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to delete loop region"]
                }

        except Exception as e:
            logger.error(f"Error deleting loop region: {e}")
            return {
                "success": False,
                "errors": [f"Delete loop error: {str(e)}"]
            }

    async def get_loop_status(self) -> Dict[str, Any]:
        """
        Get current loop status

        Returns:
            Current loop status information
        """
        try:
            if not self._loop_manager:
                await self.initialize()

            status = self._loop_manager.get_loop_status()
            return {
                "success": True,
                **status
            }

        except Exception as e:
            logger.error(f"Error getting loop status: {e}")
            return {
                "success": False,
                "errors": [f"Get status error: {str(e)}"]
            }

    async def get_project_loop_regions(self, project_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get all loop regions for a project

        Args:
            project_id: Project ID

        Returns:
            Project loop regions
        """
        try:
            if not self._loop_manager:
                await self.initialize()

            # Ensure loops are loaded for project
            await self._loop_manager.load_project_loops(project_id)

            regions = self._loop_manager.get_loaded_regions()

            return {
                "success": True,
                "project_id": str(project_id),
                "loop_regions": [self._loop_region_to_dict(region) for region in regions],
                "count": len(regions)
            }

        except Exception as e:
            logger.error(f"Error getting project loop regions: {e}")
            return {
                "success": False,
                "errors": [f"Get regions error: {str(e)}"]
            }

    def _loop_region_to_dict(self, region: LoopRegionData) -> Dict[str, Any]:
        """Convert LoopRegionData to dictionary"""
        return {
            "id": str(region.id),
            "project_id": str(region.project_id),
            "name": region.name,
            "start_time": region.start_time,
            "end_time": region.end_time,
            "duration": region.end_time - region.start_time,
            "is_enabled": region.is_enabled,
            "repeat_count": region.repeat_count,
            "auto_punch_record": region.auto_punch_record,
            "color": region.color,
            "notes": region.notes
        }

    async def _on_loop_start(self, loop_region: LoopRegionData, iteration: int):
        """Callback for loop start events"""
        try:
            if self._websocket_callback:
                message = {
                    "type": "loop_started",
                    "data": {
                        "loop_id": str(loop_region.id),
                        "loop_name": loop_region.name,
                        "iteration": iteration,
                        "start_time": loop_region.start_time
                    }
                }
                await self._websocket_callback(None, message)  # Broadcast to all

        except Exception as e:
            logger.error(f"Error in loop start callback: {e}")

    async def _on_loop_end(self, loop_region: LoopRegionData, iteration: int):
        """Callback for loop end events"""
        try:
            if self._websocket_callback:
                message = {
                    "type": "loop_ended",
                    "data": {
                        "loop_id": str(loop_region.id),
                        "loop_name": loop_region.name,
                        "iteration": iteration,
                        "end_time": loop_region.end_time
                    }
                }
                await self._websocket_callback(None, message)  # Broadcast to all

        except Exception as e:
            logger.error(f"Error in loop end callback: {e}")

    async def cleanup(self):
        """Cleanup loop service resources"""
        try:
            if self._loop_manager:
                self._loop_manager.cleanup()

            logger.info("LoopService cleanup completed")

        except Exception as e:
            logger.error(f"Error during LoopService cleanup: {e}")


# Global instance for application use
_loop_service: Optional[LoopService] = None


async def get_loop_service() -> LoopService:
    """
    Get singleton LoopService instance

    Returns:
        Global LoopService instance
    """
    global _loop_service
    if _loop_service is None:
        _loop_service = LoopService()
        await _loop_service.initialize()
    return _loop_service


async def cleanup_loop_service():
    """Cleanup loop service resources"""
    global _loop_service
    if _loop_service is not None:
        await _loop_service.cleanup()
        _loop_service = None
    logger.info("Loop service cleanup completed")