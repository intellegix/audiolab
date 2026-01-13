"""
AudioLab Recording Session Management
Real-time recording session lifecycle and audio file management
"""

import uuid
import asyncio
import tempfile
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from decimal import Decimal
from datetime import datetime
from enum import Enum

import numpy as np
import soundfile as sf
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import RecordingSession as RecordingSessionModel, Track, Clip
from ..database.connection import get_session
from .audio_input import AudioInputDevice, AudioInputManager
from ..core.config import settings

logger = logging.getLogger(__name__)


class RecordingStatus(Enum):
    """Recording session status"""
    RECORDING = "recording"
    STOPPED = "stopped"
    SAVED = "saved"
    ERROR = "error"


class AudioRecordingSession:
    """
    Individual recording session - manages audio capture and file creation
    """

    def __init__(self, session_id: uuid.UUID, track_id: uuid.UUID,
                 input_device_id: str, start_time: Decimal = Decimal("0.0")):
        """
        Initialize recording session

        Args:
            session_id: Unique session identifier
            track_id: Target track ID
            input_device_id: Audio input device ID
            start_time: Timeline position where recording starts (seconds)
        """
        self.session_id = session_id
        self.track_id = track_id
        self.input_device_id = input_device_id
        self.start_time = start_time

        # Recording state
        self.status = RecordingStatus.RECORDING
        self.audio_device: Optional[AudioInputDevice] = None
        self.audio_buffer: List[np.ndarray] = []
        self.duration: Decimal = Decimal("0.0")

        # File management
        self.temp_file_path: Optional[str] = None
        self.temp_file_handle = None
        self.final_clip_id: Optional[uuid.UUID] = None

        # Audio parameters
        self.sample_rate = 48000
        self.channels = 1
        self.bit_depth = 24

        # Callbacks
        self.progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
        self.error_callback: Optional[Callable[[Exception], None]] = None

        # Database session for async operations
        self._db_session: Optional[AsyncSession] = None

        logger.info(f"AudioRecordingSession created: {session_id} on track {track_id}")

    async def initialize_database_session(self) -> bool:
        """Initialize database session for recording operations"""
        try:
            self._db_session = await get_session()
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database session: {e}")
            return False

    async def start_recording(self, audio_input_manager: AudioInputManager) -> bool:
        """
        Start recording session

        Args:
            audio_input_manager: Audio input manager instance

        Returns:
            True if recording started successfully
        """
        try:
            # Initialize database session
            if not await self.initialize_database_session():
                return False

            # Get audio input device
            self.audio_device = await audio_input_manager.get_device(self.input_device_id)
            if not self.audio_device:
                logger.error(f"Audio device {self.input_device_id} not available")
                return False

            # Create temporary file for recording
            self.temp_file_path = await self._create_temp_audio_file()
            if not self.temp_file_path:
                return False

            # Update audio parameters from device
            self.sample_rate = self.audio_device.sample_rate
            self.channels = self.audio_device.channels

            # Start audio recording
            success = await self.audio_device.start_recording(
                callback=self._process_audio_chunk,
                session_id=str(self.session_id)
            )

            if success:
                # Update database with recording session
                await self._save_session_to_database()
                self.status = RecordingStatus.RECORDING
                logger.info(f"Recording started: session {self.session_id}")

                # Send progress update
                if self.progress_callback:
                    await self.progress_callback(self._get_progress_data())

                return True
            else:
                await self._cleanup_temp_file()
                return False

        except Exception as e:
            logger.error(f"Failed to start recording session {self.session_id}: {e}")
            if self.error_callback:
                await self.error_callback(e)
            return False

    async def stop_recording(self) -> bool:
        """
        Stop recording and finalize audio file

        Returns:
            True if stopped successfully
        """
        if self.status != RecordingStatus.RECORDING:
            logger.warning(f"Recording session {self.session_id} not currently recording")
            return False

        try:
            # Stop audio device
            if self.audio_device:
                await self.audio_device.stop_recording()

            # Close temporary file
            if self.temp_file_handle:
                self.temp_file_handle.close()
                self.temp_file_handle = None

            # Calculate final duration
            if self.audio_buffer:
                total_samples = sum(chunk.shape[0] for chunk in self.audio_buffer)
                self.duration = Decimal(str(total_samples / self.sample_rate))

            self.status = RecordingStatus.STOPPED
            logger.info(f"Recording stopped: session {self.session_id}, duration: {self.duration}s")

            # Update database
            await self._update_session_in_database()

            # Send progress update
            if self.progress_callback:
                await self.progress_callback(self._get_progress_data())

            return True

        except Exception as e:
            logger.error(f"Error stopping recording session {self.session_id}: {e}")
            self.status = RecordingStatus.ERROR
            if self.error_callback:
                await self.error_callback(e)
            return False

    async def save_as_clip(self) -> Optional[uuid.UUID]:
        """
        Save recording as audio clip in project

        Returns:
            Clip ID if saved successfully, None otherwise
        """
        if self.status not in [RecordingStatus.STOPPED, RecordingStatus.RECORDING]:
            logger.error(f"Cannot save recording session {self.session_id} in status {self.status}")
            return None

        try:
            # Stop recording if still active
            if self.status == RecordingStatus.RECORDING:
                await self.stop_recording()

            # Ensure we have recorded audio
            if not self.temp_file_path or not os.path.exists(self.temp_file_path):
                logger.error(f"No recorded audio file found for session {self.session_id}")
                return None

            if self.duration <= 0:
                logger.error(f"No recorded audio data for session {self.session_id}")
                return None

            # Create final audio file path
            final_file_path = await self._create_final_audio_file()
            if not final_file_path:
                return None

            # Move temporary file to final location
            os.rename(self.temp_file_path, final_file_path)
            self.temp_file_path = final_file_path

            # Create clip in database
            clip_id = await self._create_audio_clip(final_file_path)
            if not clip_id:
                return None

            self.final_clip_id = clip_id
            self.status = RecordingStatus.SAVED

            # Update database with final clip reference
            await self._update_session_in_database()

            logger.info(f"Recording session {self.session_id} saved as clip {clip_id}")

            # Send final progress update
            if self.progress_callback:
                await self.progress_callback(self._get_progress_data())

            return clip_id

        except Exception as e:
            logger.error(f"Error saving recording session {self.session_id} as clip: {e}")
            self.status = RecordingStatus.ERROR
            if self.error_callback:
                await self.error_callback(e)
            return None

    async def _process_audio_chunk(self, audio_data: np.ndarray, timestamp: float):
        """Process incoming audio chunk from device"""
        try:
            # Store audio data in buffer for duration calculation
            self.audio_buffer.append(audio_data.copy())

            # Write to temporary file
            if self.temp_file_handle:
                # Ensure audio data is in correct format for writing
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)

                # Write audio chunk to file
                sf.write(self.temp_file_handle, audio_data, self.sample_rate, format='WAV', subtype='FLOAT')

            # Send periodic progress updates
            if len(self.audio_buffer) % 10 == 0:  # Every 10 chunks
                current_samples = sum(chunk.shape[0] for chunk in self.audio_buffer)
                current_duration = current_samples / self.sample_rate

                if self.progress_callback:
                    progress_data = self._get_progress_data()
                    progress_data["current_duration"] = current_duration
                    await self.progress_callback(progress_data)

        except Exception as e:
            logger.error(f"Error processing audio chunk in session {self.session_id}: {e}")
            self.status = RecordingStatus.ERROR
            if self.error_callback:
                await self.error_callback(e)

    async def _create_temp_audio_file(self) -> Optional[str]:
        """Create temporary file for recording audio"""
        try:
            # Create temporary directory if needed
            temp_dir = Path(settings.AUDIO_OUTPUT_PATH) / "recordings" / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Create unique temporary file
            temp_file = tempfile.NamedTemporaryFile(
                suffix=".wav",
                prefix=f"recording_{self.session_id}_",
                dir=str(temp_dir),
                delete=False
            )

            self.temp_file_handle = temp_file
            temp_path = temp_file.name

            logger.debug(f"Created temporary recording file: {temp_path}")
            return temp_path

        except Exception as e:
            logger.error(f"Failed to create temporary recording file: {e}")
            return None

    async def _create_final_audio_file(self) -> Optional[str]:
        """Create final audio file path"""
        try:
            # Create final directory
            final_dir = Path(settings.AUDIO_OUTPUT_PATH) / "clips"
            final_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}_{self.session_id}.wav"
            final_path = final_dir / filename

            logger.debug(f"Final recording file path: {final_path}")
            return str(final_path)

        except Exception as e:
            logger.error(f"Failed to create final recording file path: {e}")
            return None

    async def _create_audio_clip(self, file_path: str) -> Optional[uuid.UUID]:
        """Create audio clip record in database"""
        try:
            if not self._db_session:
                logger.error("No database session available")
                return None

            # Create new clip
            clip_id = uuid.uuid4()
            new_clip = Clip(
                id=clip_id,
                track_id=self.track_id,
                name=f"Recording {datetime.now().strftime('%H:%M:%S')}",
                file_path=file_path,
                start_time=self.start_time,
                duration=self.duration,
                sample_rate=self.sample_rate,
                channels=self.channels,
                bit_depth=self.bit_depth,
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0
            )

            self._db_session.add(new_clip)
            await self._db_session.commit()

            logger.info(f"Created audio clip {clip_id} from recording session {self.session_id}")
            return clip_id

        except Exception as e:
            logger.error(f"Failed to create audio clip: {e}")
            if self._db_session:
                await self._db_session.rollback()
            return None

    async def _save_session_to_database(self):
        """Save recording session to database"""
        try:
            if not self._db_session:
                return

            session_record = RecordingSessionModel(
                id=self.session_id,
                track_id=self.track_id,
                input_device_id=self.input_device_id,
                start_time=self.start_time,
                status=self.status.value,
                sample_rate=self.sample_rate,
                channels=self.channels,
                bit_depth=self.bit_depth,
                temp_file_path=self.temp_file_path
            )

            self._db_session.add(session_record)
            await self._db_session.commit()

        except Exception as e:
            logger.error(f"Failed to save recording session to database: {e}")
            if self._db_session:
                await self._db_session.rollback()

    async def _update_session_in_database(self):
        """Update recording session in database"""
        try:
            if not self._db_session:
                return

            stmt = (
                update(RecordingSessionModel)
                .where(RecordingSessionModel.id == self.session_id)
                .values(
                    status=self.status.value,
                    duration=self.duration,
                    final_clip_id=self.final_clip_id,
                    updated_at=datetime.utcnow()
                )
            )

            await self._db_session.execute(stmt)
            await self._db_session.commit()

        except Exception as e:
            logger.error(f"Failed to update recording session in database: {e}")
            if self._db_session:
                await self._db_session.rollback()

    async def _cleanup_temp_file(self):
        """Cleanup temporary recording file"""
        try:
            if self.temp_file_handle:
                self.temp_file_handle.close()
                self.temp_file_handle = None

            if self.temp_file_path and os.path.exists(self.temp_file_path):
                os.unlink(self.temp_file_path)
                self.temp_file_path = None

            logger.debug(f"Cleaned up temporary files for session {self.session_id}")

        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")

    def _get_progress_data(self) -> Dict[str, Any]:
        """Get progress data for callbacks"""
        current_samples = sum(chunk.shape[0] for chunk in self.audio_buffer) if self.audio_buffer else 0
        current_duration = current_samples / self.sample_rate if self.sample_rate > 0 else 0

        return {
            "session_id": str(self.session_id),
            "track_id": str(self.track_id),
            "status": self.status.value,
            "duration": float(current_duration),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "clip_id": str(self.final_clip_id) if self.final_clip_id else None
        }

    async def cleanup(self):
        """Full cleanup of recording session"""
        try:
            # Stop recording if active
            if self.status == RecordingStatus.RECORDING:
                await self.stop_recording()

            # Cleanup temporary files
            await self._cleanup_temp_file()

            # Close database session
            if self._db_session:
                await self._db_session.close()
                self._db_session = None

            logger.debug(f"Recording session {self.session_id} cleanup completed")

        except Exception as e:
            logger.error(f"Error during recording session cleanup: {e}")


class RecordingSessionManager:
    """
    High-level recording session manager
    Manages multiple concurrent recording sessions
    """

    def __init__(self):
        """Initialize recording session manager"""
        self._active_sessions: Dict[uuid.UUID, AudioRecordingSession] = {}
        self._audio_input_manager: Optional[AudioInputManager] = None

        logger.info("RecordingSessionManager initialized")

    async def initialize(self):
        """Initialize audio input manager"""
        from .audio_input import get_audio_input_manager
        self._audio_input_manager = await get_audio_input_manager()

    async def start_recording_session(self,
                                    track_id: uuid.UUID,
                                    input_device_id: str,
                                    start_time: Decimal = Decimal("0.0"),
                                    progress_callback: Optional[Callable] = None,
                                    error_callback: Optional[Callable] = None) -> Optional[uuid.UUID]:
        """
        Start new recording session

        Args:
            track_id: Target track ID
            input_device_id: Audio input device ID
            start_time: Timeline position to start recording
            progress_callback: Progress update callback
            error_callback: Error callback

        Returns:
            Recording session ID if started successfully
        """
        try:
            if not self._audio_input_manager:
                await self.initialize()

            # Generate unique session ID
            session_id = uuid.uuid4()

            # Create recording session
            session = AudioRecordingSession(
                session_id=session_id,
                track_id=track_id,
                input_device_id=input_device_id,
                start_time=start_time
            )

            # Set callbacks
            if progress_callback:
                session.progress_callback = progress_callback
            if error_callback:
                session.error_callback = error_callback

            # Start recording
            success = await session.start_recording(self._audio_input_manager)
            if success:
                self._active_sessions[session_id] = session
                logger.info(f"Recording session {session_id} started on track {track_id}")
                return session_id
            else:
                await session.cleanup()
                return None

        except Exception as e:
            logger.error(f"Failed to start recording session: {e}")
            return None

    async def stop_recording_session(self, session_id: uuid.UUID) -> bool:
        """
        Stop recording session

        Args:
            session_id: Recording session ID

        Returns:
            True if stopped successfully
        """
        if session_id not in self._active_sessions:
            logger.warning(f"Recording session {session_id} not found")
            return False

        try:
            session = self._active_sessions[session_id]
            success = await session.stop_recording()
            logger.info(f"Recording session {session_id} stopped")
            return success

        except Exception as e:
            logger.error(f"Error stopping recording session {session_id}: {e}")
            return False

    async def save_recording_session(self, session_id: uuid.UUID) -> Optional[uuid.UUID]:
        """
        Save recording session as audio clip

        Args:
            session_id: Recording session ID

        Returns:
            Clip ID if saved successfully
        """
        if session_id not in self._active_sessions:
            logger.warning(f"Recording session {session_id} not found")
            return None

        try:
            session = self._active_sessions[session_id]
            clip_id = await session.save_as_clip()

            # Remove from active sessions after saving
            if clip_id:
                await session.cleanup()
                del self._active_sessions[session_id]
                logger.info(f"Recording session {session_id} saved and removed from active sessions")

            return clip_id

        except Exception as e:
            logger.error(f"Error saving recording session {session_id}: {e}")
            return None

    async def get_session_status(self, session_id: uuid.UUID) -> Optional[Dict[str, Any]]:
        """Get recording session status"""
        if session_id in self._active_sessions:
            session = self._active_sessions[session_id]
            return session._get_progress_data()
        return None

    async def get_all_active_sessions(self) -> List[Dict[str, Any]]:
        """Get status of all active recording sessions"""
        return [session._get_progress_data() for session in self._active_sessions.values()]

    async def stop_all_sessions(self) -> bool:
        """Stop all active recording sessions"""
        success = True
        for session_id in list(self._active_sessions.keys()):
            result = await self.stop_recording_session(session_id)
            success = success and result

        logger.info(f"Stopped all recording sessions (success: {success})")
        return success

    async def cleanup_all_sessions(self):
        """Cleanup all recording sessions"""
        for session in self._active_sessions.values():
            await session.cleanup()
        self._active_sessions.clear()
        logger.info("All recording sessions cleaned up")


# Global instance for application use
_recording_session_manager: Optional[RecordingSessionManager] = None


async def get_recording_session_manager() -> RecordingSessionManager:
    """
    Get singleton RecordingSessionManager instance

    Returns:
        Global RecordingSessionManager instance
    """
    global _recording_session_manager
    if _recording_session_manager is None:
        _recording_session_manager = RecordingSessionManager()
        await _recording_session_manager.initialize()
    return _recording_session_manager


async def cleanup_recording_sessions():
    """Cleanup recording session resources"""
    global _recording_session_manager
    if _recording_session_manager is not None:
        await _recording_session_manager.cleanup_all_sessions()
        _recording_session_manager = None
    logger.info("Recording session cleanup completed")