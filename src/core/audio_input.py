"""
AudioLab Audio Input Management
Real-time audio input device enumeration and streaming for recording
"""

import asyncio
import uuid
from typing import Callable, List, Optional, Dict, Any
import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    pyaudio = None
    PYAUDIO_AVAILABLE = False

from .environment import get_environment_detector, is_cloud_deployment, has_audio_hardware

logger = logging.getLogger(__name__)


class AudioInputStatus(Enum):
    """Audio input device status"""
    IDLE = "idle"
    RECORDING = "recording"
    ERROR = "error"


@dataclass
class AudioInputInfo:
    """Audio input device information"""
    device_id: str
    name: str
    channels: int
    sample_rate: int
    max_input_channels: int
    default_sample_rate: float
    is_default: bool = False
    host_api: str = ""


class AudioInputDevice:
    """Individual audio input device management"""

    def __init__(self, device_info: AudioInputInfo, sample_rate: int = 48000,
                 chunk_size: int = 1024, channels: int = 1):
        """
        Initialize audio input device

        Args:
            device_info: Device information from PyAudio
            sample_rate: Target sample rate (default 48kHz for professional audio)
            chunk_size: Audio buffer size in samples
            channels: Number of input channels (1=mono, 2=stereo)
        """
        self.device_info = device_info
        self.device_id = device_info.device_id
        self.name = device_info.name
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels

        self.stream: Optional[pyaudio.Stream] = None
        self.status = AudioInputStatus.IDLE
        self.callback: Optional[Callable[[np.ndarray, float], None]] = None
        self.session_id: Optional[str] = None

        # Audio processing
        # Initialize PyAudio only if available and not in cloud mode
        if PYAUDIO_AVAILABLE and has_audio_hardware():
            self._pyaudio = pyaudio.PyAudio()
        else:
            self._pyaudio = None

        self._loop = None
        self._recording_task = None

        logger.info(f"Audio input device initialized: {self.name} ({self.device_id})")

    async def start_recording(self, callback: Callable[[np.ndarray, float], None],
                            session_id: Optional[str] = None) -> bool:
        """
        Start real-time audio input stream

        Args:
            callback: Function to process audio chunks (audio_data, timestamp)
            session_id: Recording session identifier

        Returns:
            True if recording started successfully
        """
        if self.status == AudioInputStatus.RECORDING:
            logger.warning(f"Device {self.device_id} already recording")
            return False

        try:
            self.callback = callback
            self.session_id = session_id or str(uuid.uuid4())

            # Check if we have real audio hardware
            if self._pyaudio and has_audio_hardware():
                # Real audio recording
                success = await self._start_real_audio_recording()
            else:
                # Mock audio recording for cloud deployment
                success = await self._start_mock_audio_recording()

            if success:
                logger.info(f"Recording started on device {self.name} (session: {self.session_id})")
                return True
            else:
                self.status = AudioInputStatus.ERROR
                return False

        except Exception as e:
            logger.error(f"Failed to start recording on device {self.device_id}: {e}")
            self.status = AudioInputStatus.ERROR
            return False

    async def _start_real_audio_recording(self) -> bool:
        """Start real PyAudio recording for desktop environment"""
        try:
            # Configure audio stream
            self.stream = self._pyaudio.open(
                format=pyaudio.paInt24,  # 24-bit for professional audio
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=int(self.device_id),
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )

            self.stream.start_stream()
            self.status = AudioInputStatus.RECORDING

            # Start async processing loop
            self._loop = asyncio.get_event_loop()
            self._recording_task = self._loop.create_task(self._process_audio_loop())

            return True
        except Exception as e:
            logger.error(f"Real audio recording failed: {e}")
            return False

    async def _start_mock_audio_recording(self) -> bool:
        """Start mock audio recording for cloud environment"""
        try:
            # Simulate recording with generated audio data
            self.status = AudioInputStatus.RECORDING

            # Start mock recording loop
            self._loop = asyncio.get_event_loop()
            self._recording_task = self._loop.create_task(self._mock_recording_loop())

            logger.info(f"Started mock recording (cloud mode) on device {self.name}")
            return True
        except Exception as e:
            logger.error(f"Mock audio recording failed: {e}")
            return False

    async def _mock_recording_loop(self):
        """Mock recording loop for cloud environments"""
        try:
            chunk_duration = self.chunk_size / self.sample_rate  # Duration in seconds

            while self.status == AudioInputStatus.RECORDING:
                # Generate silent audio data (could be enhanced with test tones)
                mock_audio = np.zeros(self.chunk_size, dtype=np.float32)

                # Add some very quiet test tone for demonstration
                if self.channels == 1:
                    t = np.arange(self.chunk_size) / self.sample_rate
                    mock_audio = 0.001 * np.sin(2 * np.pi * 440 * t).astype(np.float32)  # Quiet 440Hz tone

                # If stereo, duplicate the signal
                if self.channels > 1:
                    mock_audio = mock_audio.reshape(-1, 1)
                    mock_audio = np.tile(mock_audio, (1, self.channels))

                # Call the callback with mock data
                current_time = asyncio.get_event_loop().time()
                if self.callback:
                    await self.callback(mock_audio, current_time)

                # Wait for the chunk duration to simulate real-time
                await asyncio.sleep(chunk_duration)

        except asyncio.CancelledError:
            logger.debug("Mock recording loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Mock recording loop error: {e}")
            self.status = AudioInputStatus.ERROR

    async def stop_recording(self) -> bool:
        """
        Stop audio input and cleanup

        Returns:
            True if stopped successfully
        """
        if self.status != AudioInputStatus.RECORDING:
            logger.warning(f"Device {self.device_id} not currently recording")
            return False

        try:
            # Stop the recording task
            if self._recording_task:
                self._recording_task.cancel()
                try:
                    await self._recording_task
                except asyncio.CancelledError:
                    pass

            # Stop and close stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            self.status = AudioInputStatus.IDLE
            self.callback = None
            self.session_id = None

            logger.info(f"Recording stopped on device {self.name}")
            return True

        except Exception as e:
            logger.error(f"Error stopping recording on device {self.device_id}: {e}")
            self.status = AudioInputStatus.ERROR
            return False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        PyAudio stream callback - processes incoming audio data
        """
        if status:
            logger.warning(f"Audio input status flag: {status}")

        # Convert bytes to numpy array
        audio_data = np.frombuffer(in_data, dtype=np.int32)  # 24-bit as int32

        # Normalize to float32 [-1.0, 1.0]
        audio_data = audio_data.astype(np.float32) / (2**23)  # 24-bit normalization

        # Reshape for multi-channel
        if self.channels > 1:
            audio_data = audio_data.reshape(-1, self.channels)

        # Get current timestamp
        current_time = time_info['input_buffer_adc_time']

        # Queue for async processing
        if self.callback and self._loop:
            self._loop.call_soon_threadsafe(
                self._queue_audio_data, audio_data.copy(), current_time
            )

        return (None, pyaudio.paContinue)

    def _queue_audio_data(self, audio_data: np.ndarray, timestamp: float):
        """Queue audio data for async processing"""
        if self._loop and self.callback:
            self._loop.create_task(self._process_audio_chunk(audio_data, timestamp))

    async def _process_audio_chunk(self, audio_data: np.ndarray, timestamp: float):
        """Process individual audio chunk asynchronously"""
        try:
            if self.callback:
                await self.callback(audio_data, timestamp)
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    async def _process_audio_loop(self):
        """Main audio processing loop"""
        try:
            while self.status == AudioInputStatus.RECORDING:
                await asyncio.sleep(0.001)  # Small delay to prevent CPU spinning
        except asyncio.CancelledError:
            logger.debug("Audio processing loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Audio processing loop error: {e}")
            self.status = AudioInputStatus.ERROR

    def get_status(self) -> Dict[str, Any]:
        """Get current device status"""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "status": self.status.value,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "session_id": self.session_id,
            "is_recording": self.status == AudioInputStatus.RECORDING
        }

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_pyaudio'):
            try:
                self._pyaudio.terminate()
            except:
                pass


class AudioInputManager:
    """High-level audio input device management"""

    def __init__(self):
        """Initialize audio input manager"""
        # Initialize PyAudio only if available and not in cloud mode
        if PYAUDIO_AVAILABLE and has_audio_hardware():
            self._pyaudio = pyaudio.PyAudio()
        else:
            self._pyaudio = None

        self._devices: Dict[str, AudioInputDevice] = {}
        self._available_devices: List[AudioInputInfo] = []

        # Refresh device list on initialization
        asyncio.create_task(self.refresh_device_list())

        logger.info("AudioInputManager initialized")

    async def enumerate_devices(self) -> List[AudioInputInfo]:
        """
        List available audio input devices

        Returns:
            List of AudioInputInfo objects
        """
        try:
            # Check if we have real audio hardware
            if self._pyaudio and has_audio_hardware():
                devices = await self._enumerate_real_devices()
            else:
                devices = await self._enumerate_mock_devices()

            self._available_devices = devices
            logger.info(f"Found {len(devices)} audio input devices")

            return devices

        except Exception as e:
            logger.error(f"Failed to enumerate audio devices: {e}")
            return []

    async def _enumerate_real_devices(self) -> List[AudioInputInfo]:
        """Enumerate real audio devices using PyAudio"""
        try:
            devices = []
            device_count = self._pyaudio.get_device_count()
            default_device = self._pyaudio.get_default_input_device_info()

            for i in range(device_count):
                try:
                    device_info = self._pyaudio.get_device_info_by_index(i)

                    # Only include input devices
                    if device_info['maxInputChannels'] > 0:
                        host_api_info = self._pyaudio.get_host_api_info_by_index(
                            device_info['hostApi']
                        )

                        audio_info = AudioInputInfo(
                            device_id=str(i),
                            name=device_info['name'],
                            channels=device_info['maxInputChannels'],
                            sample_rate=48000,  # Target sample rate
                            max_input_channels=device_info['maxInputChannels'],
                            default_sample_rate=device_info['defaultSampleRate'],
                            is_default=(i == default_device['index']),
                            host_api=host_api_info['name']
                        )

                        devices.append(audio_info)

                except Exception as e:
                    logger.warning(f"Error getting device info for index {i}: {e}")
                    continue

            return devices

        except Exception as e:
            logger.error(f"Failed to enumerate real audio devices: {e}")
            return []

    async def _enumerate_mock_devices(self) -> List[AudioInputInfo]:
        """Create mock audio devices for cloud deployment"""
        try:
            env_detector = get_environment_detector()
            cloud_platform = env_detector._detection_cache.get('cloud_platform', 'cloud')

            mock_devices = [
                AudioInputInfo(
                    device_id="mock_0",
                    name=f"AudioLab Virtual Microphone 1 ({cloud_platform.title()})",
                    channels=1,
                    sample_rate=48000,
                    max_input_channels=1,
                    default_sample_rate=48000.0,
                    is_default=True,
                    host_api="Virtual Audio"
                ),
                AudioInputInfo(
                    device_id="mock_1",
                    name=f"AudioLab Virtual Microphone 2 ({cloud_platform.title()})",
                    channels=2,
                    sample_rate=48000,
                    max_input_channels=2,
                    default_sample_rate=48000.0,
                    is_default=False,
                    host_api="Virtual Audio"
                ),
                AudioInputInfo(
                    device_id="mock_2",
                    name=f"AudioLab Virtual Line In ({cloud_platform.title()})",
                    channels=2,
                    sample_rate=48000,
                    max_input_channels=2,
                    default_sample_rate=48000.0,
                    is_default=False,
                    host_api="Virtual Audio"
                )
            ]

            logger.info(f"Created {len(mock_devices)} mock audio devices for cloud deployment")
            return mock_devices

        except Exception as e:
            logger.error(f"Failed to create mock audio devices: {e}")
            return []

    async def get_device(self, device_id: str) -> Optional[AudioInputDevice]:
        """
        Get specific input device by ID

        Args:
            device_id: Device identifier

        Returns:
            AudioInputDevice instance or None if not found
        """
        # Check if device already instantiated
        if device_id in self._devices:
            return self._devices[device_id]

        # Find device info
        device_info = None
        for info in self._available_devices:
            if info.device_id == device_id:
                device_info = info
                break

        if not device_info:
            # Refresh device list and try again
            await self.refresh_device_list()
            for info in self._available_devices:
                if info.device_id == device_id:
                    device_info = info
                    break

        if not device_info:
            logger.warning(f"Audio device {device_id} not found")
            return None

        # Create device instance
        try:
            device = AudioInputDevice(device_info)
            self._devices[device_id] = device
            return device
        except Exception as e:
            logger.error(f"Failed to create audio input device {device_id}: {e}")
            return None

    async def get_default_device(self) -> Optional[AudioInputDevice]:
        """
        Get default audio input device

        Returns:
            Default AudioInputDevice or None if not found
        """
        try:
            # Refresh devices to ensure we have current list
            await self.refresh_device_list()

            # Find default device
            for info in self._available_devices:
                if info.is_default:
                    return await self.get_device(info.device_id)

            # Fallback to first available device
            if self._available_devices:
                return await self.get_device(self._available_devices[0].device_id)

            logger.warning("No audio input devices available")
            return None

        except Exception as e:
            logger.error(f"Failed to get default audio device: {e}")
            return None

    async def refresh_device_list(self):
        """Refresh the list of available devices"""
        await self.enumerate_devices()

    def get_device_status(self, device_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of specific device

        Args:
            device_id: Device identifier

        Returns:
            Device status dict or None if not found
        """
        if device_id in self._devices:
            return self._devices[device_id].get_status()
        return None

    def get_all_device_status(self) -> List[Dict[str, Any]]:
        """Get status of all instantiated devices"""
        return [device.get_status() for device in self._devices.values()]

    async def stop_all_recording(self) -> bool:
        """
        Stop recording on all devices

        Returns:
            True if all devices stopped successfully
        """
        success = True
        for device in self._devices.values():
            if device.status == AudioInputStatus.RECORDING:
                result = await device.stop_recording()
                success = success and result

        logger.info(f"Stopped recording on all devices (success: {success})")
        return success

    def __del__(self):
        """Cleanup on destruction"""
        if hasattr(self, '_pyaudio'):
            try:
                self._pyaudio.terminate()
            except:
                pass


# Global instance for application use
_audio_input_manager: Optional[AudioInputManager] = None


async def get_audio_input_manager() -> AudioInputManager:
    """
    Get singleton AudioInputManager instance

    Returns:
        Global AudioInputManager instance
    """
    global _audio_input_manager
    if _audio_input_manager is None:
        _audio_input_manager = AudioInputManager()
        await _audio_input_manager.enumerate_devices()
    return _audio_input_manager


async def cleanup_audio_input():
    """Cleanup audio input resources"""
    global _audio_input_manager
    if _audio_input_manager is not None:
        await _audio_input_manager.stop_all_recording()
        _audio_input_manager = None
    logger.info("Audio input cleanup completed")