"""
AudioLab Environment Detection
Detects runtime environment and configures appropriate audio capabilities
"""

import os
import platform
import logging
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Environment types for AudioLab deployment"""
    DESKTOP = "desktop"      # Local desktop with audio hardware
    CLOUD = "cloud"          # Cloud deployment (Render, AWS, etc.)
    CONTAINER = "container"  # Docker container
    CI_CD = "ci_cd"         # Continuous integration environment


class AudioCapability(Enum):
    """Audio capability levels"""
    FULL = "full"           # Real-time I/O, recording, playback
    PROCESSING_ONLY = "processing_only"  # File processing only
    MOCK = "mock"           # Mock audio for testing
    DISABLED = "disabled"   # No audio capabilities


class EnvironmentDetector:
    """
    Detects the current runtime environment and determines audio capabilities
    """

    def __init__(self):
        """Initialize environment detector"""
        self._environment_type: Optional[EnvironmentType] = None
        self._audio_capability: Optional[AudioCapability] = None
        self._detection_cache: Dict[str, Any] = {}

        # Perform detection
        self._detect_environment()
        logger.info(f"Environment detected: {self._environment_type.value}, Audio: {self._audio_capability.value}")

    def _detect_environment(self):
        """Detect the current environment type and capabilities"""

        # Check for cloud deployment indicators
        if self._is_cloud_environment():
            self._environment_type = EnvironmentType.CLOUD
            self._audio_capability = AudioCapability.PROCESSING_ONLY
            return

        # Check for container environment
        if self._is_container_environment():
            self._environment_type = EnvironmentType.CONTAINER
            self._audio_capability = self._detect_container_audio()
            return

        # Check for CI/CD environment
        if self._is_ci_cd_environment():
            self._environment_type = EnvironmentType.CI_CD
            self._audio_capability = AudioCapability.MOCK
            return

        # Default to desktop environment
        self._environment_type = EnvironmentType.DESKTOP
        self._audio_capability = self._detect_desktop_audio()

    def _is_cloud_environment(self) -> bool:
        """Detect if running in cloud environment (Render, AWS, etc.)"""
        cloud_indicators = [
            # Render.com indicators
            os.environ.get('RENDER'),
            os.environ.get('RENDER_SERVICE_ID'),
            os.environ.get('RENDER_SERVICE_NAME') == 'audiolab-api',

            # General cloud indicators
            os.environ.get('PORT'),  # Cloud platforms set PORT env var
            os.environ.get('DYNO'),  # Heroku
            os.environ.get('AWS_EXECUTION_ENV'),  # AWS Lambda
            os.environ.get('GCLOUD_PROJECT'),  # Google Cloud
            os.environ.get('WEBSITE_SITE_NAME'),  # Azure

            # Check if running on cloud domains
            os.environ.get('RENDER_EXTERNAL_URL', '').endswith('.onrender.com'),
        ]

        is_cloud = any(cloud_indicators)
        if is_cloud:
            self._detection_cache['cloud_platform'] = self._identify_cloud_platform()

        return is_cloud

    def _is_container_environment(self) -> bool:
        """Detect if running in a container (Docker, etc.)"""
        container_indicators = [
            os.path.exists('/.dockerenv'),
            os.path.exists('/proc/1/cgroup') and 'docker' in open('/proc/1/cgroup', 'r').read(),
            os.environ.get('CONTAINER') == 'true',
            platform.system() == 'Linux' and os.path.exists('/proc/self/mountinfo')
        ]

        try:
            # Check for container-specific indicators
            return any(container_indicators)
        except Exception:
            return False

    def _is_ci_cd_environment(self) -> bool:
        """Detect if running in CI/CD environment"""
        ci_cd_indicators = [
            os.environ.get('CI') == 'true',
            os.environ.get('CONTINUOUS_INTEGRATION') == 'true',
            os.environ.get('GITHUB_ACTIONS') == 'true',
            os.environ.get('GITLAB_CI') == 'true',
            os.environ.get('JENKINS_URL'),
            os.environ.get('TRAVIS') == 'true',
            os.environ.get('CIRCLECI') == 'true'
        ]

        return any(ci_cd_indicators)

    def _identify_cloud_platform(self) -> str:
        """Identify specific cloud platform"""
        if os.environ.get('RENDER_SERVICE_ID'):
            return 'render'
        elif os.environ.get('DYNO'):
            return 'heroku'
        elif os.environ.get('AWS_EXECUTION_ENV'):
            return 'aws'
        elif os.environ.get('GCLOUD_PROJECT'):
            return 'gcp'
        elif os.environ.get('WEBSITE_SITE_NAME'):
            return 'azure'
        else:
            return 'unknown_cloud'

    def _detect_desktop_audio(self) -> AudioCapability:
        """Detect audio capabilities on desktop environment"""
        try:
            # Try to import PyAudio to check for audio system
            import pyaudio

            # Try to initialize PyAudio
            pa = pyaudio.PyAudio()
            device_count = pa.get_device_count()
            pa.terminate()

            if device_count > 0:
                self._detection_cache['audio_devices'] = device_count
                return AudioCapability.FULL
            else:
                logger.warning("PyAudio initialized but no audio devices found")
                return AudioCapability.PROCESSING_ONLY

        except ImportError:
            logger.warning("PyAudio not available - audio processing only")
            return AudioCapability.PROCESSING_ONLY
        except Exception as e:
            logger.warning(f"Audio system detection failed: {e}")
            return AudioCapability.PROCESSING_ONLY

    def _detect_container_audio(self) -> AudioCapability:
        """Detect audio capabilities in container environment"""
        # Containers typically don't have audio hardware access
        # but might have audio processing capabilities
        try:
            import pyaudio
            # If PyAudio is available, we can at least do processing
            return AudioCapability.PROCESSING_ONLY
        except ImportError:
            return AudioCapability.MOCK

    def get_environment_type(self) -> EnvironmentType:
        """Get detected environment type"""
        return self._environment_type

    def get_audio_capability(self) -> AudioCapability:
        """Get detected audio capability level"""
        return self._audio_capability

    def is_cloud_deployment(self) -> bool:
        """Check if running in cloud deployment"""
        return self._environment_type == EnvironmentType.CLOUD

    def is_desktop_environment(self) -> bool:
        """Check if running on desktop"""
        return self._environment_type == EnvironmentType.DESKTOP

    def has_real_audio_hardware(self) -> bool:
        """Check if real audio hardware is available"""
        return self._audio_capability == AudioCapability.FULL

    def can_process_audio_files(self) -> bool:
        """Check if audio file processing is available"""
        return self._audio_capability in [
            AudioCapability.FULL,
            AudioCapability.PROCESSING_ONLY
        ]

    def should_use_mock_audio(self) -> bool:
        """Check if mock audio should be used"""
        return self._audio_capability in [AudioCapability.MOCK, AudioCapability.DISABLED]

    def get_recommended_audio_config(self) -> Dict[str, Any]:
        """Get recommended audio configuration for current environment"""
        if self._environment_type == EnvironmentType.CLOUD:
            return {
                "sample_rate": 48000,
                "bit_depth": 24,
                "channels": 2,
                "buffer_size": 1024,
                "real_time_recording": False,
                "file_based_recording": True,
                "mock_devices": True,
                "enable_websocket_audio": True
            }
        elif self._environment_type == EnvironmentType.DESKTOP:
            return {
                "sample_rate": 48000,
                "bit_depth": 24,
                "channels": 2,
                "buffer_size": 512,  # Lower latency for desktop
                "real_time_recording": True,
                "file_based_recording": True,
                "mock_devices": False,
                "enable_websocket_audio": True
            }
        else:
            return {
                "sample_rate": 48000,
                "bit_depth": 24,
                "channels": 2,
                "buffer_size": 1024,
                "real_time_recording": False,
                "file_based_recording": True,
                "mock_devices": True,
                "enable_websocket_audio": False
            }

    def get_environment_info(self) -> Dict[str, Any]:
        """Get comprehensive environment information"""
        info = {
            "environment_type": self._environment_type.value,
            "audio_capability": self._audio_capability.value,
            "platform": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python_version": platform.python_version()
            },
            "runtime": {
                "is_cloud": self.is_cloud_deployment(),
                "is_desktop": self.is_desktop_environment(),
                "has_audio_hardware": self.has_real_audio_hardware(),
                "can_process_files": self.can_process_audio_files()
            }
        }

        # Add cached detection details
        info.update(self._detection_cache)

        return info

    def get_audio_service_config(self) -> Dict[str, Any]:
        """Get configuration for audio services"""
        return {
            "enable_real_time_recording": self.has_real_audio_hardware(),
            "enable_file_based_recording": True,
            "enable_playback_engine": self.has_real_audio_hardware(),
            "enable_mock_devices": self.should_use_mock_audio(),
            "audio_config": self.get_recommended_audio_config()
        }


# Global environment detector instance
_environment_detector: Optional[EnvironmentDetector] = None


def get_environment_detector() -> EnvironmentDetector:
    """
    Get singleton EnvironmentDetector instance

    Returns:
        Global EnvironmentDetector instance
    """
    global _environment_detector
    if _environment_detector is None:
        _environment_detector = EnvironmentDetector()
    return _environment_detector


def is_cloud_deployment() -> bool:
    """Quick check if running in cloud deployment"""
    return get_environment_detector().is_cloud_deployment()


def has_audio_hardware() -> bool:
    """Quick check if audio hardware is available"""
    return get_environment_detector().has_real_audio_hardware()


def get_audio_config() -> Dict[str, Any]:
    """Quick access to recommended audio configuration"""
    return get_environment_detector().get_recommended_audio_config()