"""
AudioLab Recording Service
High-level recording orchestration and API integration
"""

import uuid
import logging
from typing import List, Optional, Dict, Any, Callable
from decimal import Decimal
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import Track, Project, RecordingSession
from ..database.connection import get_session
from ..core.audio_input import get_audio_input_manager, AudioInputInfo
from ..core.recording_session import get_recording_session_manager, RecordingSessionManager

logger = logging.getLogger(__name__)


class RecordingService:
    """
    High-level recording service for orchestrating audio recording operations
    Integrates audio input management with database operations
    """

    def __init__(self):
        """Initialize recording service"""
        self._recording_manager: Optional[RecordingSessionManager] = None

        # WebSocket callback for real-time updates
        self._websocket_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

        logger.info("RecordingService initialized")

    async def initialize(self):
        """Initialize recording service dependencies"""
        try:
            self._recording_manager = await get_recording_session_manager()
            logger.info("RecordingService dependencies initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RecordingService: {e}")
            raise

    def set_websocket_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Set WebSocket callback for real-time progress updates

        Args:
            callback: Function to send WebSocket messages (connection_id, message)
        """
        self._websocket_callback = callback

    async def get_available_audio_devices(self) -> List[AudioInputInfo]:
        """
        Get list of available audio input devices

        Returns:
            List of AudioInputInfo objects
        """
        try:
            audio_manager = await get_audio_input_manager()
            devices = await audio_manager.enumerate_devices()

            logger.info(f"Retrieved {len(devices)} available audio devices")
            return devices

        except Exception as e:
            logger.error(f"Error getting available audio devices: {e}")
            return []

    async def validate_recording_request(self, track_id: uuid.UUID, device_id: str) -> Dict[str, Any]:
        """
        Validate recording request parameters

        Args:
            track_id: Target track ID
            device_id: Audio input device ID

        Returns:
            Validation result with success status and details
        """
        result = {
            "valid": True,
            "errors": [],
            "track": None,
            "device": None
        }

        try:
            # Validate track exists and is record-enabled
            async with get_session() as db:
                stmt = select(Track).where(Track.id == track_id)
                track_result = await db.execute(stmt)
                track = track_result.scalar_one_or_none()

                if not track:
                    result["valid"] = False
                    result["errors"].append(f"Track {track_id} not found")
                else:
                    result["track"] = {
                        "id": str(track.id),
                        "name": track.name,
                        "record_enabled": track.record_enabled,
                        "input_device_id": track.input_device_id
                    }

                    if not track.record_enabled:
                        result["valid"] = False
                        result["errors"].append(f"Track '{track.name}' is not record-enabled")

            # Validate audio device is available
            audio_manager = await get_audio_input_manager()
            device = await audio_manager.get_device(device_id)

            if not device:
                result["valid"] = False
                result["errors"].append(f"Audio device {device_id} not available")
            else:
                result["device"] = device.get_status()

                # Check if device is already in use
                if device.get_status()["is_recording"]:
                    result["valid"] = False
                    result["errors"].append(f"Audio device {device_id} is already recording")

            logger.info(f"Recording validation for track {track_id}, device {device_id}: {'VALID' if result['valid'] else 'INVALID'}")
            return result

        except Exception as e:
            logger.error(f"Error validating recording request: {e}")
            result["valid"] = False
            result["errors"].append(f"Validation error: {str(e)}")
            return result

    async def start_recording(self,
                            track_id: uuid.UUID,
                            device_id: str,
                            start_time: Decimal = Decimal("0.0"),
                            connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Start recording on specified track

        Args:
            track_id: Target track ID
            device_id: Audio input device ID
            start_time: Timeline position to start recording (seconds)
            connection_id: WebSocket connection ID for progress updates

        Returns:
            Recording session information
        """
        try:
            if not self._recording_manager:
                await self.initialize()

            # Validate request
            validation = await self.validate_recording_request(track_id, device_id)
            if not validation["valid"]:
                return {
                    "success": False,
                    "errors": validation["errors"],
                    "session_id": None
                }

            # Create progress callback for WebSocket updates
            async def progress_callback(progress_data: Dict[str, Any]):
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "recording_progress",
                        "data": progress_data
                    }
                    await self._websocket_callback(connection_id, message)

            # Create error callback
            async def error_callback(error: Exception):
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "recording_error",
                        "data": {
                            "session_id": str(session_id) if 'session_id' in locals() else None,
                            "error": str(error)
                        }
                    }
                    await self._websocket_callback(connection_id, message)
                logger.error(f"Recording error: {error}")

            # Start recording session
            session_id = await self._recording_manager.start_recording_session(
                track_id=track_id,
                input_device_id=device_id,
                start_time=start_time,
                progress_callback=progress_callback,
                error_callback=error_callback
            )

            if session_id:
                logger.info(f"Recording started: session {session_id}, track {track_id}, device {device_id}")

                # Send initial WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "recording_started",
                        "data": {
                            "session_id": str(session_id),
                            "track_id": str(track_id),
                            "device_id": device_id,
                            "start_time": float(start_time)
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "session_id": str(session_id),
                    "track": validation["track"],
                    "device": validation["device"]
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to start recording session"],
                    "session_id": None
                }

        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            return {
                "success": False,
                "errors": [f"Recording start error: {str(e)}"],
                "session_id": None
            }

    async def stop_recording(self,
                           session_id: uuid.UUID,
                           connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop recording session

        Args:
            session_id: Recording session ID
            connection_id: WebSocket connection ID for updates

        Returns:
            Stop operation result
        """
        try:
            if not self._recording_manager:
                await self.initialize()

            # Stop recording
            success = await self._recording_manager.stop_recording_session(session_id)

            if success:
                logger.info(f"Recording stopped: session {session_id}")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    session_status = await self._recording_manager.get_session_status(session_id)
                    message = {
                        "type": "recording_stopped",
                        "data": {
                            "session_id": str(session_id),
                            "status": session_status
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "session_id": str(session_id)
                }
            else:
                return {
                    "success": False,
                    "errors": [f"Failed to stop recording session {session_id}"]
                }

        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
            return {
                "success": False,
                "errors": [f"Recording stop error: {str(e)}"]
            }

    async def save_recording(self,
                           session_id: uuid.UUID,
                           connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Save recording session as audio clip

        Args:
            session_id: Recording session ID
            connection_id: WebSocket connection ID for updates

        Returns:
            Save operation result with clip ID
        """
        try:
            if not self._recording_manager:
                await self.initialize()

            # Save recording as clip
            clip_id = await self._recording_manager.save_recording_session(session_id)

            if clip_id:
                logger.info(f"Recording saved: session {session_id}, clip {clip_id}")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "recording_saved",
                        "data": {
                            "session_id": str(session_id),
                            "clip_id": str(clip_id)
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "session_id": str(session_id),
                    "clip_id": str(clip_id)
                }
            else:
                return {
                    "success": False,
                    "errors": [f"Failed to save recording session {session_id}"]
                }

        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            return {
                "success": False,
                "errors": [f"Recording save error: {str(e)}"]
            }

    async def stop_and_save_recording(self,
                                    session_id: uuid.UUID,
                                    connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop recording and immediately save as clip

        Args:
            session_id: Recording session ID
            connection_id: WebSocket connection ID for updates

        Returns:
            Combined stop and save operation result
        """
        try:
            # Stop recording first
            stop_result = await self.stop_recording(session_id, connection_id)
            if not stop_result["success"]:
                return stop_result

            # Save recording
            save_result = await self.save_recording(session_id, connection_id)

            return {
                "success": save_result["success"],
                "session_id": str(session_id),
                "clip_id": save_result.get("clip_id"),
                "errors": save_result.get("errors", [])
            }

        except Exception as e:
            logger.error(f"Error stopping and saving recording: {e}")
            return {
                "success": False,
                "errors": [f"Stop and save error: {str(e)}"]
            }

    async def get_recording_status(self, session_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """
        Get status of recording session

        Args:
            session_id: Recording session ID

        Returns:
            Recording session status
        """
        try:
            if not self._recording_manager:
                await self.initialize()

            return await self._recording_manager.get_session_status(session_id)

        except Exception as e:
            logger.error(f"Error getting recording status: {e}")
            return None

    async def get_all_active_recordings(self) -> List[Dict[str, Any]]:
        """
        Get status of all active recording sessions

        Returns:
            List of active recording session statuses
        """
        try:
            if not self._recording_manager:
                await self.initialize()

            return await self._recording_manager.get_all_active_sessions()

        except Exception as e:
            logger.error(f"Error getting active recordings: {e}")
            return []

    async def stop_all_recordings(self, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop all active recording sessions

        Args:
            connection_id: WebSocket connection ID for updates

        Returns:
            Stop all operation result
        """
        try:
            if not self._recording_manager:
                await self.initialize()

            # Get active sessions before stopping
            active_sessions = await self._recording_manager.get_all_active_sessions()

            # Stop all sessions
            success = await self._recording_manager.stop_all_sessions()

            if success:
                logger.info("All recording sessions stopped")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "all_recordings_stopped",
                        "data": {
                            "stopped_sessions": [session["session_id"] for session in active_sessions]
                        }
                    }
                    await self._websocket_callback(connection_id, message)

            return {
                "success": success,
                "stopped_sessions": len(active_sessions)
            }

        except Exception as e:
            logger.error(f"Error stopping all recordings: {e}")
            return {
                "success": False,
                "errors": [f"Stop all error: {str(e)}"]
            }

    async def enable_track_recording(self, track_id: uuid.UUID, device_id: str) -> Dict[str, Any]:
        """
        Enable recording on a track with specified input device

        Args:
            track_id: Track ID to enable recording
            device_id: Audio input device ID

        Returns:
            Enable operation result
        """
        try:
            async with get_session() as db:
                # Get track
                stmt = select(Track).where(Track.id == track_id)
                result = await db.execute(stmt)
                track = result.scalar_one_or_none()

                if not track:
                    return {
                        "success": False,
                        "errors": [f"Track {track_id} not found"]
                    }

                # Validate device is available
                audio_manager = await get_audio_input_manager()
                device = await audio_manager.get_device(device_id)

                if not device:
                    return {
                        "success": False,
                        "errors": [f"Audio device {device_id} not available"]
                    }

                # Update track recording settings
                track.record_enabled = True
                track.input_device_id = device_id

                await db.commit()

                logger.info(f"Recording enabled for track {track_id} with device {device_id}")

                return {
                    "success": True,
                    "track_id": str(track_id),
                    "device_id": device_id,
                    "track_name": track.name
                }

        except Exception as e:
            logger.error(f"Error enabling track recording: {e}")
            return {
                "success": False,
                "errors": [f"Enable recording error: {str(e)}"]
            }

    async def disable_track_recording(self, track_id: uuid.UUID) -> Dict[str, Any]:
        """
        Disable recording on a track

        Args:
            track_id: Track ID to disable recording

        Returns:
            Disable operation result
        """
        try:
            async with get_session() as db:
                # Get track
                stmt = select(Track).where(Track.id == track_id)
                result = await db.execute(stmt)
                track = result.scalar_one_or_none()

                if not track:
                    return {
                        "success": False,
                        "errors": [f"Track {track_id} not found"]
                    }

                # Update track recording settings
                track.record_enabled = False
                track.input_device_id = None

                await db.commit()

                logger.info(f"Recording disabled for track {track_id}")

                return {
                    "success": True,
                    "track_id": str(track_id),
                    "track_name": track.name
                }

        except Exception as e:
            logger.error(f"Error disabling track recording: {e}")
            return {
                "success": False,
                "errors": [f"Disable recording error: {str(e)}"]
            }


# Global instance for application use
_recording_service: Optional[RecordingService] = None


async def get_recording_service() -> RecordingService:
    """
    Get singleton RecordingService instance

    Returns:
        Global RecordingService instance
    """
    global _recording_service
    if _recording_service is None:
        _recording_service = RecordingService()
        await _recording_service.initialize()
    return _recording_service