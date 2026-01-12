"""
AudioLab Testing Configuration
Pytest fixtures and test setup
"""
import pytest
import asyncio
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing"""
    import numpy as np

    # 1 second of stereo audio at 48kHz
    sample_rate = 48000
    duration = 1.0
    samples = int(sample_rate * duration)

    # Generate sine wave test audio (440 Hz A note)
    t = np.linspace(0, duration, samples)
    left_channel = np.sin(2 * np.pi * 440 * t)
    right_channel = np.sin(2 * np.pi * 440 * t * 1.01)  # Slightly detuned

    # Combine to stereo
    audio = np.column_stack((left_channel, right_channel)).astype(np.float32)

    return {
        "audio": audio,
        "sample_rate": sample_rate,
        "duration": duration,
        "channels": 2
    }


@pytest.fixture
def test_audio_config():
    """Test audio configuration"""
    return {
        "sample_rate": 48000,
        "buffer_size": 512,
        "channels": 2,
        "latency_target_ms": 10.0,
        "max_tracks": 32
    }


@pytest.fixture
def test_project_data():
    """Test project data"""
    return {
        "name": "Test Project",
        "sample_rate": 48000,
        "bit_depth": 24,
        "tempo": 120.0,
        "time_signature": "4/4"
    }


@pytest.fixture
def mock_demucs_result():
    """Mock Demucs separation result"""
    import numpy as np

    # Create mock separated stems
    audio_length = 48000  # 1 second at 48kHz
    stems = {
        "vocals": np.random.randn(audio_length, 2).astype(np.float32) * 0.5,
        "drums": np.random.randn(audio_length, 2).astype(np.float32) * 0.3,
        "bass": np.random.randn(audio_length, 2).astype(np.float32) * 0.4,
        "other": np.random.randn(audio_length, 2).astype(np.float32) * 0.2
    }

    return stems


@pytest.fixture
def test_effects_config():
    """Test effects configuration"""
    return {
        "eq": {
            "bands": [
                {"freq": 100, "gain": 0, "q": 0.7, "type": "highpass"},
                {"freq": 3000, "gain": 2, "q": 1.5, "type": "bell"}
            ]
        },
        "compressor": {
            "threshold": -18.0,
            "ratio": 3.0,
            "attack": 10.0,
            "release": 100.0
        }
    }


class TestSettings:
    """Test configuration settings"""

    def __init__(self):
        self.DEFAULT_SAMPLE_RATE = 48000
        self.DEFAULT_BUFFER_SIZE = 512
        self.MAX_TRACKS = 32
        self.DATABASE_URL = "sqlite:///test.db"
        self.REDIS_URL = "redis://localhost:6379/1"  # Test database


@pytest.fixture
def test_settings():
    """Test configuration settings fixture"""
    return TestSettings()


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )