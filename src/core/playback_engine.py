"""
AudioLab Playback Engine
Synchronized multi-track audio playback with real-time position tracking
"""

import asyncio
import uuid
import logging
import threading
import time
from typing import Dict, List, Optional, Callable, Any
from decimal import Decimal
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PYAUDIO_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    SOUNDFILE_AVAILABLE = False

from ..database.models import Track, Clip, Project
from ..core.config import settings
from .environment import get_environment_detector, has_audio_hardware

logger = logging.getLogger(__name__)


class PlaybackStatus(Enum):
    """Playback engine status"""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class AudioClipData:
    """Pre-loaded audio clip data for playback"""
    clip_id: uuid.UUID
    track_id: uuid.UUID
    audio_data: np.ndarray
    start_time: float  # Timeline position in seconds
    duration: float  # Duration in seconds
    sample_rate: int
    channels: int
    volume: float = 1.0
    pan: float = 0.0  # -1.0 (left) to 1.0 (right)


@dataclass
class TrackData:
    """Track configuration for playback"""
    track_id: uuid.UUID
    track_index: int
    name: str
    volume: float
    pan: float
    muted: bool
    soloed: bool
    clips: List[AudioClipData]


class PlaybackEngine:
    """
    Core multi-track audio playback engine
    Handles synchronized playback of multiple tracks with real-time position tracking
    """

    def __init__(self, sample_rate: int = 48000, buffer_size: int = 1024, channels: int = 2):
        """
        Initialize playback engine

        Args:
            sample_rate: Audio sample rate (48kHz professional)
            buffer_size: Audio buffer size in samples
            channels: Output channels (2 for stereo)
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.channels = channels

        # Playback state
        self.status = PlaybackStatus.STOPPED
        self.current_position: float = 0.0  # Current position in seconds
        self.start_position: float = 0.0  # Position where playback started
        self.tempo: Decimal = Decimal("120.0")  # BPM
        self.project_duration: float = 0.0  # Total project duration

        # Audio data
        self.tracks: Dict[uuid.UUID, TrackData] = {}
        self.project: Optional[Project] = None

        # Audio output (only if hardware available)
        if PYAUDIO_AVAILABLE and has_audio_hardware():
            self._pyaudio = pyaudio.PyAudio()
            self._has_audio_output = True
        else:
            self._pyaudio = None
            self._has_audio_output = False

        self._output_stream: Optional[pyaudio.Stream] = None

        # Threading for real-time playback
        self._playback_thread: Optional[threading.Thread] = None
        self._playback_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

        # Position tracking
        self._position_callbacks: List[Callable[[float], None]] = []
        self._position_update_interval = 0.05  # 50ms updates
        self._last_position_update = 0.0

        # Loop management
        self.loop_manager = None  # Will be set by loop manager

        # Performance metrics
        self._playback_start_time = 0.0
        self._frames_processed = 0

        logger.info(f"PlaybackEngine initialized: {sample_rate}Hz, {buffer_size} buffer, {channels}ch")

    def load_project_data(self, project: Project, tracks: List[TrackData]):
        """
        Load project data for playback

        Args:
            project: Project model with tempo and settings
            tracks: List of TrackData with pre-loaded audio clips
        """
        try:
            self.project = project
            self.tempo = project.tempo

            # Clear existing tracks
            self.tracks.clear()

            # Load tracks
            for track_data in tracks:
                self.tracks[track_data.track_id] = track_data

            # Calculate project duration
            self.project_duration = self._calculate_project_duration()

            logger.info(f"Loaded project data: {len(tracks)} tracks, {self.project_duration:.2f}s duration")

        except Exception as e:
            logger.error(f"Error loading project data: {e}")
            raise

    def _calculate_project_duration(self) -> float:
        """Calculate total project duration from all clips"""
        max_end_time = 0.0
        for track in self.tracks.values():
            for clip in track.clips:
                clip_end_time = clip.start_time + clip.duration
                max_end_time = max(max_end_time, clip_end_time)
        return max_end_time

    async def play(self, start_position: float = 0.0) -> bool:
        """
        Start playback from specified position

        Args:
            start_position: Timeline position to start playback (seconds)

        Returns:
            True if playback started successfully
        """
        with self._playback_lock:
            if self.status == PlaybackStatus.PLAYING:
                logger.warning("Playback already active")
                return False

            try:
                self.start_position = start_position
                self.current_position = start_position

                # Reset events
                self._stop_event.clear()
                self._pause_event.clear()

                # Initialize audio output (real or mock)
                if self._has_audio_output:
                    if not self._init_audio_output():
                        return False

                    # Start real audio playback thread
                    self._playback_thread = threading.Thread(
                        target=self._playback_loop,
                        name=f"AudioPlayback-{uuid.uuid4().hex[:8]}",
                        daemon=True
                    )
                    self._playback_thread.start()
                else:
                    # Start mock playback for cloud environment
                    self._playback_thread = threading.Thread(
                        target=self._mock_playback_loop,
                        name=f"MockPlayback-{uuid.uuid4().hex[:8]}",
                        daemon=True
                    )
                    self._playback_thread.start()

                self.status = PlaybackStatus.PLAYING
                self._playback_start_time = time.time()

                logger.info(f"Playback started at position {start_position:.3f}s")
                await self._notify_position_callbacks(self.current_position)

                return True

            except Exception as e:
                logger.error(f"Failed to start playback: {e}")
                self.status = PlaybackStatus.ERROR
                return False

    async def stop(self) -> bool:
        """
        Stop playback and cleanup

        Returns:
            True if stopped successfully
        """
        try:
            if self.status == PlaybackStatus.STOPPED:
                return True

            # Signal stop
            self._stop_event.set()

            # Wait for playback thread to finish
            if self._playback_thread and self._playback_thread.is_alive():
                self._playback_thread.join(timeout=2.0)

            # Cleanup audio output
            self._cleanup_audio_output()

            with self._playback_lock:
                self.status = PlaybackStatus.STOPPED
                self.current_position = 0.0

            logger.info("Playback stopped")
            await self._notify_position_callbacks(0.0)

            return True

        except Exception as e:
            logger.error(f"Error stopping playback: {e}")
            return False

    async def pause(self) -> bool:
        """
        Pause playback

        Returns:
            True if paused successfully
        """
        if self.status != PlaybackStatus.PLAYING:
            return False

        try:
            self._pause_event.set()
            self.status = PlaybackStatus.PAUSED
            logger.info(f"Playback paused at position {self.current_position:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Error pausing playback: {e}")
            return False

    async def resume(self) -> bool:
        """
        Resume paused playback

        Returns:
            True if resumed successfully
        """
        if self.status != PlaybackStatus.PAUSED:
            return False

        try:
            self._pause_event.clear()
            self.status = PlaybackStatus.PLAYING
            logger.info(f"Playback resumed at position {self.current_position:.3f}s")
            return True

        except Exception as e:
            logger.error(f"Error resuming playback: {e}")
            return False

    async def seek_to(self, position: float) -> bool:
        """
        Jump to specific timeline position

        Args:
            position: Target position in seconds

        Returns:
            True if seek successful
        """
        try:
            # Clamp position to valid range
            position = max(0.0, min(position, self.project_duration))

            was_playing = self.status == PlaybackStatus.PLAYING

            if was_playing:
                await self.stop()

            self.current_position = position

            if was_playing:
                await self.play(position)

            logger.info(f"Seeked to position {position:.3f}s")
            await self._notify_position_callbacks(position)

            return True

        except Exception as e:
            logger.error(f"Error seeking to position {position}: {e}")
            return False

    def get_current_position(self) -> float:
        """Get current playback position in seconds"""
        return self.current_position

    def get_project_duration(self) -> float:
        """Get total project duration in seconds"""
        return self.project_duration

    def add_position_callback(self, callback: Callable[[float], None]):
        """
        Add callback for position updates

        Args:
            callback: Function to call with position updates
        """
        if callback not in self._position_callbacks:
            self._position_callbacks.append(callback)

    def remove_position_callback(self, callback: Callable[[float], None]):
        """Remove position update callback"""
        if callback in self._position_callbacks:
            self._position_callbacks.remove(callback)

    def set_loop_manager(self, loop_manager):
        """
        Set loop manager for automatic looping functionality

        Args:
            loop_manager: LoopManager instance
        """
        self.loop_manager = loop_manager

    def set_track_volume(self, track_id: uuid.UUID, volume: float):
        """Set track volume (0.0 to 2.0)"""
        if track_id in self.tracks:
            self.tracks[track_id].volume = max(0.0, min(volume, 2.0))

    def set_track_mute(self, track_id: uuid.UUID, muted: bool):
        """Set track mute state"""
        if track_id in self.tracks:
            self.tracks[track_id].muted = muted

    def set_track_solo(self, track_id: uuid.UUID, soloed: bool):
        """Set track solo state"""
        if track_id in self.tracks:
            self.tracks[track_id].soloed = soloed

    def get_playback_info(self) -> Dict[str, Any]:
        """Get current playback information"""
        return {
            "status": self.status.value,
            "position": self.current_position,
            "duration": self.project_duration,
            "tempo": float(self.tempo),
            "sample_rate": self.sample_rate,
            "tracks_loaded": len(self.tracks),
            "frames_processed": self._frames_processed
        }

    def _init_audio_output(self) -> bool:
        """Initialize PyAudio output stream"""
        try:
            self._output_stream = self._pyaudio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback
            )

            self._output_stream.start_stream()
            return True

        except Exception as e:
            logger.error(f"Failed to initialize audio output: {e}")
            return False

    def _cleanup_audio_output(self):
        """Cleanup audio output stream"""
        try:
            if self._output_stream:
                self._output_stream.stop_stream()
                self._output_stream.close()
                self._output_stream = None

        except Exception as e:
            logger.error(f"Error cleaning up audio output: {e}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio stream callback for real-time audio output"""
        if status:
            logger.warning(f"Audio output status: {status}")

        try:
            # Check if we should stop or pause
            if self._stop_event.is_set():
                return (np.zeros((frame_count, self.channels), dtype=np.float32), pyaudio.paComplete)

            if self._pause_event.is_set():
                return (np.zeros((frame_count, self.channels), dtype=np.float32), pyaudio.paContinue)

            # Generate audio buffer
            audio_buffer = self._generate_audio_buffer(frame_count)

            # Update position
            duration_seconds = frame_count / self.sample_rate
            self.current_position += duration_seconds
            self._frames_processed += frame_count

            # Check loop boundaries
            if self.loop_manager:
                loop_restarted = asyncio.run_coroutine_threadsafe(
                    self.loop_manager.check_loop_boundary(self.current_position),
                    asyncio.get_event_loop()
                ).result()

                # If loop restarted, get the new position
                if loop_restarted:
                    self.current_position = self.loop_manager.last_position

            # Check if we've reached the end (only if no active loop)
            if (self.current_position >= self.project_duration and
                (not self.loop_manager or not self.loop_manager.active_loop)):
                self._stop_event.set()
                return (audio_buffer, pyaudio.paComplete)

            return (audio_buffer, pyaudio.paContinue)

        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            self._stop_event.set()
            return (np.zeros((frame_count, self.channels), dtype=np.float32), pyaudio.paComplete)

    def _generate_audio_buffer(self, frame_count: int) -> np.ndarray:
        """
        Generate mixed audio buffer for current position

        Args:
            frame_count: Number of frames to generate

        Returns:
            Mixed audio buffer
        """
        try:
            # Initialize output buffer
            output_buffer = np.zeros((frame_count, self.channels), dtype=np.float32)

            # Check for solo tracks
            has_solo = any(track.soloed for track in self.tracks.values())

            # Mix each track
            for track in self.tracks.values():
                # Skip muted tracks
                if track.muted:
                    continue

                # Skip non-solo tracks when solo is active
                if has_solo and not track.soloed:
                    continue

                # Mix track audio
                track_buffer = self._generate_track_buffer(track, frame_count)
                if track_buffer is not None:
                    output_buffer += track_buffer

            # Apply master volume (could be configurable)
            master_volume = 0.8  # Prevent clipping
            output_buffer *= master_volume

            # Clip to prevent distortion
            output_buffer = np.clip(output_buffer, -1.0, 1.0)

            return output_buffer

        except Exception as e:
            logger.error(f"Error generating audio buffer: {e}")
            return np.zeros((frame_count, self.channels), dtype=np.float32)

    def _generate_track_buffer(self, track: TrackData, frame_count: int) -> Optional[np.ndarray]:
        """
        Generate audio buffer for a single track

        Args:
            track: Track data
            frame_count: Number of frames to generate

        Returns:
            Track audio buffer or None if no audio
        """
        try:
            track_buffer = np.zeros((frame_count, self.channels), dtype=np.float32)
            has_audio = False

            # Check each clip in the track
            for clip in track.clips:
                clip_audio = self._get_clip_audio(clip, frame_count)
                if clip_audio is not None:
                    track_buffer += clip_audio
                    has_audio = True

            if not has_audio:
                return None

            # Apply track volume
            track_buffer *= track.volume

            # Apply panning (simple stereo panning)
            if self.channels == 2 and track.pan != 0.0:
                left_gain = 1.0 - max(0.0, track.pan)
                right_gain = 1.0 + min(0.0, track.pan)
                track_buffer[:, 0] *= left_gain  # Left channel
                track_buffer[:, 1] *= right_gain  # Right channel

            return track_buffer

        except Exception as e:
            logger.error(f"Error generating track buffer for {track.track_id}: {e}")
            return None

    def _get_clip_audio(self, clip: AudioClipData, frame_count: int) -> Optional[np.ndarray]:
        """
        Get audio data from clip for current position

        Args:
            clip: Audio clip data
            frame_count: Number of frames needed

        Returns:
            Clip audio buffer or None if no audio at current position
        """
        try:
            # Check if current position overlaps with clip
            clip_end_time = clip.start_time + clip.duration
            if (self.current_position < clip.start_time or
                self.current_position >= clip_end_time):
                return None

            # Calculate position within clip
            position_in_clip = self.current_position - clip.start_time
            start_sample = int(position_in_clip * clip.sample_rate)

            # Calculate how many samples to read
            duration_seconds = frame_count / self.sample_rate
            samples_needed = int(duration_seconds * clip.sample_rate)

            # Ensure we don't read past end of clip
            clip_total_samples = len(clip.audio_data)
            end_sample = min(start_sample + samples_needed, clip_total_samples)

            if start_sample >= clip_total_samples:
                return None

            # Extract audio segment
            audio_segment = clip.audio_data[start_sample:end_sample]

            if len(audio_segment) == 0:
                return None

            # Resample if necessary (simple approach)
            if clip.sample_rate != self.sample_rate:
                # Simple resampling - for production, use librosa.resample
                ratio = self.sample_rate / clip.sample_rate
                new_length = int(len(audio_segment) * ratio)
                audio_segment = np.interp(
                    np.linspace(0, len(audio_segment) - 1, new_length),
                    np.arange(len(audio_segment)),
                    audio_segment
                )

            # Convert to output format
            if audio_segment.ndim == 1:
                # Mono to stereo if needed
                if self.channels == 2:
                    audio_segment = np.column_stack([audio_segment, audio_segment])
                else:
                    audio_segment = audio_segment.reshape(-1, 1)
            elif audio_segment.shape[1] == 1 and self.channels == 2:
                # Mono to stereo
                audio_segment = np.column_stack([audio_segment[:, 0], audio_segment[:, 0]])

            # Pad or trim to exact frame count
            if len(audio_segment) < frame_count:
                padding = np.zeros((frame_count - len(audio_segment), self.channels), dtype=np.float32)
                audio_segment = np.vstack([audio_segment, padding])
            elif len(audio_segment) > frame_count:
                audio_segment = audio_segment[:frame_count]

            # Apply clip volume
            audio_segment *= clip.volume

            return audio_segment.astype(np.float32)

        except Exception as e:
            logger.error(f"Error getting clip audio for {clip.clip_id}: {e}")
            return None

    def _playback_loop(self):
        """Main playback thread loop for position updates (real audio)"""
        try:
            while not self._stop_event.is_set():
                # Send position updates
                if (time.time() - self._last_position_update) >= self._position_update_interval:
                    asyncio.run_coroutine_threadsafe(
                        self._notify_position_callbacks(self.current_position),
                        asyncio.get_event_loop()
                    )
                    self._last_position_update = time.time()

                # Small sleep to prevent CPU spinning
                time.sleep(0.01)

        except Exception as e:
            logger.error(f"Error in playback loop: {e}")
        finally:
            logger.debug("Playback loop exited")

    def _mock_playback_loop(self):
        """Mock playback thread loop for cloud environments"""
        try:
            logger.info("Starting mock playback loop (cloud mode)")

            while not self._stop_event.is_set():
                if self._pause_event.is_set():
                    time.sleep(0.1)
                    continue

                # Simulate audio processing timing
                frame_duration = self.buffer_size / self.sample_rate
                self.current_position += frame_duration
                self._frames_processed += self.buffer_size

                # Check loop boundaries
                if self.loop_manager:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop_restarted = loop.run_until_complete(
                            self.loop_manager.check_loop_boundary(self.current_position)
                        )

                        # If loop restarted, get the new position
                        if loop_restarted:
                            self.current_position = self.loop_manager.last_position

                        loop.close()
                    except Exception as e:
                        logger.warning(f"Mock playback loop boundary check error: {e}")

                # Check if we've reached the end (only if no active loop)
                if (self.current_position >= self.project_duration and
                    (not self.loop_manager or not self.loop_manager.active_loop)):
                    self._stop_event.set()
                    break

                # Send position updates
                if (time.time() - self._last_position_update) >= self._position_update_interval:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(
                            self._notify_position_callbacks(self.current_position)
                        )
                        self._last_position_update = time.time()
                        loop.close()
                    except Exception as e:
                        logger.warning(f"Mock playback position update error: {e}")

                # Sleep for frame duration to simulate real-time
                time.sleep(frame_duration)

        except Exception as e:
            logger.error(f"Error in mock playback loop: {e}")
        finally:
            logger.debug("Mock playback loop exited")

    async def _notify_position_callbacks(self, position: float):
        """Notify all position callbacks with current position"""
        for callback in self._position_callbacks[:]:  # Copy list to avoid modification during iteration
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(position)
                else:
                    callback(position)
            except Exception as e:
                logger.error(f"Error in position callback: {e}")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            if hasattr(self, '_stop_event'):
                self._stop_event.set()

            if hasattr(self, '_output_stream') and self._output_stream:
                self._cleanup_audio_output()

            if hasattr(self, '_pyaudio'):
                self._pyaudio.terminate()

        except Exception as e:
            logger.error(f"Error in PlaybackEngine destructor: {e}")


# Global instance for application use
_playback_engine: Optional[PlaybackEngine] = None


def get_playback_engine() -> PlaybackEngine:
    """
    Get singleton PlaybackEngine instance

    Returns:
        Global PlaybackEngine instance
    """
    global _playback_engine
    if _playback_engine is None:
        _playback_engine = PlaybackEngine()
    return _playback_engine


async def cleanup_playback_engine():
    """Cleanup playback engine resources"""
    global _playback_engine
    if _playback_engine is not None:
        await _playback_engine.stop()
        _playback_engine = None
    logger.info("Playback engine cleanup completed")