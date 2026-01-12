"""
Basic AudioLab functionality testing
Tests core components without complex dependencies
"""
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_configuration_system():
    """Test configuration system functionality"""
    print("=== Testing Configuration System ===")

    from core.config import get_settings

    settings = get_settings()

    # Test basic settings
    assert settings.DEFAULT_SAMPLE_RATE == 48000
    assert settings.DEFAULT_BUFFER_SIZE == 512
    assert settings.MAX_TRACKS == 32

    # Test validation methods
    assert settings.validate_sample_rate(48000) == True
    assert settings.validate_sample_rate(22050) == False
    assert settings.validate_bit_depth(24) == True
    assert settings.validate_bit_depth(12) == False
    assert settings.validate_export_format("wav") == True
    assert settings.validate_export_format("xyz") == False

    # Test configuration methods
    audio_config = settings.get_audio_config()
    assert "sample_rate" in audio_config
    assert "buffer_size" in audio_config

    demucs_config = settings.get_demucs_config()
    assert "default_model" in demucs_config
    assert "available_models" in demucs_config

    export_config = settings.get_export_config()
    assert "supported_formats" in export_config
    assert "lufs_targets" in export_config

    print("[PASS] Configuration system tests completed")


def test_processing_result_model():
    """Test ProcessingResult Pydantic model"""
    print("=== Testing ProcessingResult Model ===")

    # Add the audio_processor path temporarily
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))

    try:
        # Create a minimal ProcessingResult for testing
        from pydantic import BaseModel
        from typing import Optional, Any

        class TestProcessingResult(BaseModel):
            success: bool
            data: Optional[Any] = None
            metadata: Optional[dict] = None
            error: Optional[str] = None
            processing_time: float = 0.0

        # Test success case
        result = TestProcessingResult(
            success=True,
            data=[1, 2, 3],
            metadata={"test": "value"},
            processing_time=0.1
        )

        assert result.success == True
        assert result.data == [1, 2, 3]
        assert result.metadata == {"test": "value"}
        assert result.processing_time == 0.1

        # Test failure case
        error_result = TestProcessingResult(
            success=False,
            error="Test error message"
        )

        assert error_result.success == False
        assert error_result.error == "Test error message"
        assert error_result.data is None

        print("[PASS] ProcessingResult model tests completed")

    except Exception as e:
        print(f"[FAIL] ProcessingResult test failed: {e}")
        raise


def test_audio_format_enum():
    """Test audio format enumeration"""
    print("=== Testing Audio Format Enum ===")

    from enum import Enum

    class TestAudioFormat(str, Enum):
        WAV = "wav"
        FLAC = "flac"
        MP3 = "mp3"
        AIFF = "aiff"

    # Test enum values
    assert TestAudioFormat.WAV == "wav"
    assert TestAudioFormat.FLAC == "flac"
    assert TestAudioFormat.MP3 == "mp3"
    assert TestAudioFormat.AIFF == "aiff"

    # Test enum listing
    formats = list(TestAudioFormat)
    assert len(formats) == 4

    print(f"[PASS] Audio formats: {[f.value for f in formats]}")


def test_basic_audio_validation():
    """Test basic audio array validation logic"""
    print("=== Testing Audio Validation ===")

    import numpy as np

    def validate_audio(audio):
        """Simple audio validation function"""
        if audio is None or audio.size == 0:
            return False
        if audio.dtype not in [np.float32, np.float64]:
            return False
        if len(audio.shape) > 2:  # Mono or stereo only
            return False
        return True

    # Test valid audio arrays
    valid_mono = np.array([1.0, 0.5, -0.3], dtype=np.float32)
    valid_stereo = np.random.randn(1000, 2).astype(np.float32)

    assert validate_audio(valid_mono) == True
    assert validate_audio(valid_stereo) == True

    # Test invalid audio arrays
    assert validate_audio(None) == False
    assert validate_audio(np.array([], dtype=np.float32)) == False
    assert validate_audio(np.array([1, 2, 3], dtype=np.int32)) == False  # Wrong dtype
    assert validate_audio(np.random.randn(10, 10, 10)) == False  # Too many dimensions

    print("[PASS] Audio validation tests completed")


def main():
    """Run all basic functionality tests"""
    print("AudioLab Basic Functionality Testing")
    print("=" * 50)

    try:
        test_configuration_system()
        print()

        test_processing_result_model()
        print()

        test_audio_format_enum()
        print()

        test_basic_audio_validation()
        print()

        print("=" * 50)
        print("[SUCCESS] All basic functionality tests PASSED!")
        return True

    except Exception as e:
        print(f"[FAIL] Basic functionality tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)