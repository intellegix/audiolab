"""
AudioLab Playback Service
High-level playback orchestration with database integration
"""

import uuid
import asyncio
import logging
import os
from typing import List, Optional, Dict, Any, Callable
from decimal import Decimal

import numpy as np
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    SOUNDFILE_AVAILABLE = False
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database.models import Project, Track, Clip
from ..database.connection import get_session
from ..core.playback_engine import (
    PlaybackEngine, get_playback_engine, TrackData, AudioClipData
)

logger = logging.getLogger(__name__)


class PlaybackService:
    """
    High-level playback service for orchestrating multi-track audio playback
    Integrates playback engine with database and provides API-level functionality
    """

    def __init__(self):
        """Initialize playback service"""
        self._playback_engine: Optional[PlaybackEngine] = None
        self._current_project_id: Optional[uuid.UUID] = None
        self._loaded_tracks: Dict[uuid.UUID, TrackData] = {}

        # WebSocket callback for real-time updates
        self._websocket_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

        # Position update tracking
        self._position_update_task: Optional[asyncio.Task] = None

        logger.info("PlaybackService initialized")

    async def initialize(self):
        """Initialize playback service dependencies"""
        try:
            self._playback_engine = get_playback_engine()

            # Add position callback for WebSocket updates
            self._playback_engine.add_position_callback(self._on_position_update)

            logger.info("PlaybackService dependencies initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PlaybackService: {e}")
            raise

    def set_websocket_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """
        Set WebSocket callback for real-time position updates

        Args:
            callback: Function to send WebSocket messages (connection_id, message)
        """
        self._websocket_callback = callback

    async def load_project(self, project_id: uuid.UUID) -> Dict[str, Any]:
        """
        Load project data for playback

        Args:
            project_id: Project ID to load

        Returns:
            Load operation result
        """
        try:
            if not self._playback_engine:
                await self.initialize()

            # Stop current playback if active
            if self._playback_engine.status.value != "stopped":
                await self._playback_engine.stop()

            async with get_session() as db:
                # Load project with all tracks and clips
                stmt = (
                    select(Project)
                    .options(
                        selectinload(Project.tracks).selectinload(Track.clips)
                    )
                    .where(Project.id == project_id)
                )

                result = await db.execute(stmt)
                project = result.scalar_one_or_none()

                if not project:
                    return {
                        "success": False,
                        "errors": [f"Project {project_id} not found"]
                    }

                # Pre-load audio clips
                tracks_data = []
                total_clips_loaded = 0

                for track in project.tracks:
                    clips_data = []

                    for clip in track.clips:
                        clip_audio = await self._load_clip_audio(clip)
                        if clip_audio:
                            clips_data.append(clip_audio)
                            total_clips_loaded += 1

                    track_data = TrackData(
                        track_id=track.id,
                        track_index=track.track_index,
                        name=track.name,
                        volume=float(track.volume),
                        pan=float(track.pan),
                        muted=track.muted,
                        soloed=track.soloed,
                        clips=clips_data
                    )
                    tracks_data.append(track_data)

                # Load data into playback engine
                self._playback_engine.load_project_data(project, tracks_data)
                self._current_project_id = project_id
                self._loaded_tracks = {track.track_id: track for track in tracks_data}

                logger.info(f"Loaded project {project_id}: {len(tracks_data)} tracks, {total_clips_loaded} clips")

                return {
                    "success": True,
                    "project_id": str(project_id),
                    "project_name": project.name,
                    "tracks_loaded": len(tracks_data),
                    "clips_loaded": total_clips_loaded,
                    "duration": self._playback_engine.get_project_duration(),
                    "tempo": float(project.tempo),
                    "sample_rate": project.sample_rate
                }

        except Exception as e:
            logger.error(f"Error loading project {project_id}: {e}")
            return {
                "success": False,
                "errors": [f"Project load error: {str(e)}"]
            }

    async def play(self, start_position: float = 0.0, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Start playback from specified position

        Args:
            start_position: Timeline position to start playback (seconds)
            connection_id: WebSocket connection ID for updates

        Returns:
            Play operation result
        """
        try:
            if not self._playback_engine:
                await self.initialize()

            if not self._current_project_id:
                return {
                    "success": False,
                    "errors": ["No project loaded for playback"]
                }

            success = await self._playback_engine.play(start_position)

            if success:
                logger.info(f"Playback started at position {start_position:.3f}s")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "playback_started",
                        "data": {
                            "position": start_position,
                            "project_id": str(self._current_project_id),
                            "playback_info": self._playback_engine.get_playback_info()
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "position": start_position,
                    "status": "playing",
                    "playback_info": self._playback_engine.get_playback_info()
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to start playback"]
                }

        except Exception as e:
            logger.error(f"Error starting playback: {e}")
            return {
                "success": False,
                "errors": [f"Playback start error: {str(e)}"]
            }

    async def stop(self, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop playback

        Args:
            connection_id: WebSocket connection ID for updates

        Returns:
            Stop operation result
        """
        try:
            if not self._playback_engine:
                return {"success": False, "errors": ["Playback engine not initialized"]}

            success = await self._playback_engine.stop()

            if success:
                logger.info("Playback stopped")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "playback_stopped",
                        "data": {
                            "position": 0.0,
                            "status": "stopped"
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "status": "stopped",
                    "position": 0.0
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to stop playback"]
                }

        except Exception as e:
            logger.error(f"Error stopping playback: {e}")
            return {
                "success": False,
                "errors": [f"Playback stop error: {str(e)}"]
            }

    async def pause(self, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Pause playback

        Args:
            connection_id: WebSocket connection ID for updates

        Returns:
            Pause operation result
        """
        try:
            if not self._playback_engine:
                return {"success": False, "errors": ["Playback engine not initialized"]}

            success = await self._playback_engine.pause()
            current_position = self._playback_engine.get_current_position()

            if success:
                logger.info(f"Playback paused at position {current_position:.3f}s")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "playback_paused",
                        "data": {
                            "position": current_position,
                            "status": "paused"
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "status": "paused",
                    "position": current_position
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to pause playback"]
                }

        except Exception as e:
            logger.error(f"Error pausing playback: {e}")
            return {
                "success": False,
                "errors": [f"Playback pause error: {str(e)}"]
            }

    async def resume(self, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Resume paused playback

        Args:
            connection_id: WebSocket connection ID for updates

        Returns:
            Resume operation result
        """
        try:
            if not self._playback_engine:
                return {"success": False, "errors": ["Playback engine not initialized"]}

            success = await self._playback_engine.resume()
            current_position = self._playback_engine.get_current_position()

            if success:
                logger.info(f"Playback resumed at position {current_position:.3f}s")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "playback_resumed",
                        "data": {
                            "position": current_position,
                            "status": "playing"
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "status": "playing",
                    "position": current_position
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to resume playback"]
                }

        except Exception as e:
            logger.error(f"Error resuming playback: {e}")
            return {
                "success": False,
                "errors": [f"Playback resume error: {str(e)}"]
            }

    async def seek(self, position: float, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Seek to specific timeline position

        Args:
            position: Target position in seconds
            connection_id: WebSocket connection ID for updates

        Returns:
            Seek operation result
        """
        try:
            if not self._playback_engine:
                return {"success": False, "errors": ["Playback engine not initialized"]}

            success = await self._playback_engine.seek_to(position)

            if success:
                logger.info(f"Playback seeked to position {position:.3f}s")

                # Send WebSocket notification
                if self._websocket_callback and connection_id:
                    message = {
                        "type": "playback_seeked",
                        "data": {
                            "position": position,
                            "status": self._playback_engine.status.value
                        }
                    }
                    await self._websocket_callback(connection_id, message)

                return {
                    "success": True,
                    "position": position,
                    "status": self._playback_engine.status.value
                }
            else:
                return {
                    "success": False,
                    "errors": ["Failed to seek to position"]
                }

        except Exception as e:
            logger.error(f"Error seeking to position {position}: {e}")
            return {
                "success": False,
                "errors": [f"Playback seek error: {str(e)}"]
            }

    async def get_playback_status(self) -> Dict[str, Any]:
        """
        Get current playback status

        Returns:
            Current playback status information
        """
        try:
            if not self._playback_engine:
                return {
                    "status": "stopped",
                    "position": 0.0,
                    "duration": 0.0,
                    "project_loaded": False
                }

            playback_info = self._playback_engine.get_playback_info()

            return {
                "status": playback_info["status"],
                "position": playback_info["position"],
                "duration": playback_info["duration"],
                "tempo": playback_info["tempo"],
                "project_id": str(self._current_project_id) if self._current_project_id else None,
                "project_loaded": self._current_project_id is not None,
                "tracks_loaded": len(self._loaded_tracks),
                "sample_rate": playback_info["sample_rate"]
            }

        except Exception as e:
            logger.error(f"Error getting playback status: {e}")
            return {
                "status": "error",
                "position": 0.0,
                "duration": 0.0,
                "project_loaded": False,
                "error": str(e)
            }

    async def set_track_volume(self, track_id: uuid.UUID, volume: float) -> Dict[str, Any]:
        """
        Set track volume

        Args:
            track_id: Track ID
            volume: Volume level (0.0 to 2.0)

        Returns:
            Set volume operation result
        """
        try:
            if not self._playback_engine:
                return {"success": False, "errors": ["Playback engine not initialized"]}

            # Validate volume range
            volume = max(0.0, min(volume, 2.0))

            # Update playback engine
            self._playback_engine.set_track_volume(track_id, volume)

            # Update database
            async with get_session() as db:
                stmt = select(Track).where(Track.id == track_id)
                result = await db.execute(stmt)
                track = result.scalar_one_or_none()

                if track:
                    track.volume = Decimal(str(volume))
                    await db.commit()

            logger.info(f"Track {track_id} volume set to {volume}")

            return {
                "success": True,
                "track_id": str(track_id),
                "volume": volume
            }

        except Exception as e:
            logger.error(f"Error setting track volume: {e}")
            return {
                "success": False,
                "errors": [f"Set volume error: {str(e)}"]
            }

    async def set_track_mute(self, track_id: uuid.UUID, muted: bool) -> Dict[str, Any]:
        """
        Set track mute state

        Args:
            track_id: Track ID
            muted: Mute state

        Returns:
            Set mute operation result
        """
        try:
            if not self._playback_engine:
                return {"success": False, "errors": ["Playback engine not initialized"]}

            # Update playback engine
            self._playback_engine.set_track_mute(track_id, muted)

            # Update database
            async with get_session() as db:
                stmt = select(Track).where(Track.id == track_id)
                result = await db.execute(stmt)
                track = result.scalar_one_or_none()

                if track:
                    track.muted = muted
                    await db.commit()

            logger.info(f"Track {track_id} mute set to {muted}")

            return {
                "success": True,
                "track_id": str(track_id),
                "muted": muted
            }

        except Exception as e:
            logger.error(f"Error setting track mute: {e}")
            return {
                "success": False,
                "errors": [f"Set mute error: {str(e)}"]
            }

    async def set_track_solo(self, track_id: uuid.UUID, soloed: bool) -> Dict[str, Any]:
        """
        Set track solo state

        Args:
            track_id: Track ID
            soloed: Solo state

        Returns:
            Set solo operation result
        """
        try:
            if not self._playback_engine:
                return {"success": False, "errors": ["Playback engine not initialized"]}

            # Update playback engine
            self._playback_engine.set_track_solo(track_id, soloed)

            # Update database
            async with get_session() as db:
                stmt = select(Track).where(Track.id == track_id)
                result = await db.execute(stmt)
                track = result.scalar_one_or_none()

                if track:
                    track.soloed = soloed
                    await db.commit()

            logger.info(f"Track {track_id} solo set to {soloed}")

            return {
                "success": True,
                "track_id": str(track_id),
                "soloed": soloed
            }

        except Exception as e:
            logger.error(f"Error setting track solo: {e}")
            return {
                "success": False,
                "errors": [f"Set solo error: {str(e)}"]
            }

    async def _load_clip_audio(self, clip: Clip) -> Optional[AudioClipData]:
        """
        Load audio data from clip file

        Args:
            clip: Clip model from database

        Returns:
            AudioClipData object with loaded audio or None if failed
        """
        try:
            if not clip.file_path or not os.path.exists(clip.file_path):
                logger.warning(f"Clip audio file not found: {clip.file_path}")
                return None

            # Load audio file
            if SOUNDFILE_AVAILABLE:
                audio_data, sample_rate = sf.read(clip.file_path, dtype='float32')
            else:
                # Fallback: create mock audio data for cloud environment
                logger.warning(f"SoundFile not available, creating mock audio for clip {clip.id}")
                duration_seconds = float(clip.duration)
                sample_rate = clip.sample_rate or 48000
                num_samples = int(duration_seconds * sample_rate)

                # Create simple test tone or silence
                audio_data = np.zeros(num_samples, dtype=np.float32)
                # Optionally add a quiet test tone
                if duration_seconds > 0:
                    t = np.arange(num_samples) / sample_rate
                    audio_data = 0.001 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

            # Ensure audio data is 2D (samples, channels)
            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, 1)

            clip_data = AudioClipData(
                clip_id=clip.id,
                track_id=clip.track_id,
                audio_data=audio_data,
                start_time=float(clip.start_time),
                duration=float(clip.duration),
                sample_rate=sample_rate,
                channels=audio_data.shape[1],
                volume=1.0,  # Clip-level volume, can be configurable
                pan=0.0      # Clip-level pan, can be configurable
            )

            return clip_data

        except Exception as e:
            logger.error(f"Error loading clip audio {clip.id}: {e}")
            return None

    async def _on_position_update(self, position: float):
        """Callback for playback position updates"""
        try:
            if self._websocket_callback:
                message = {
                    "type": "position_update",
                    "data": {
                        "position": position,
                        "project_id": str(self._current_project_id) if self._current_project_id else None
                    }
                }
                # Broadcast to all connected clients (connection_id can be None for broadcast)
                await self._websocket_callback(None, message)

        except Exception as e:
            logger.error(f"Error in position update callback: {e}")

    async def cleanup(self):
        """Cleanup playback service resources"""
        try:
            if self._playback_engine:
                await self._playback_engine.stop()

            if self._position_update_task:
                self._position_update_task.cancel()

            self._current_project_id = None
            self._loaded_tracks.clear()

            logger.info("PlaybackService cleanup completed")

        except Exception as e:
            logger.error(f"Error during PlaybackService cleanup: {e}")


# Global instance for application use
_playback_service: Optional[PlaybackService] = None


async def get_playback_service() -> PlaybackService:
    """
    Get singleton PlaybackService instance

    Returns:
        Global PlaybackService instance
    """
    global _playback_service
    if _playback_service is None:
        _playback_service = PlaybackService()
        await _playback_service.initialize()
    return _playback_service


async def cleanup_playback_service():
    """Cleanup playback service resources"""
    global _playback_service
    if _playback_service is not None:
        await _playback_service.cleanup()
        _playback_service = None
    logger.info("Playback service cleanup completed")