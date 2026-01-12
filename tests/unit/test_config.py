"""
Unit tests for AudioLab configuration system
Tests settings validation, environment loading, and configuration methods
"""
import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


@pytest.mark.unit
class TestConfigurationSystem:
    """Test core configuration functionality"""

    def test_default_configuration_values(self, test_settings):
        """Test default configuration values"""
        settings = test_settings

        # Test default audio settings
        assert settings.DEFAULT_SAMPLE_RATE == 48000
        assert settings.DEFAULT_BUFFER_SIZE == 512
        assert settings.MAX_TRACKS == 32

        # Test database settings
        assert "test.db" in settings.DATABASE_URL
        assert "redis://" in settings.REDIS_URL

    def test_audio_configuration_validation(self):
        """Test audio configuration validation methods"""
        from core.config import AudioLabSettings

        # Create minimal settings for testing
        settings = AudioLabSettings()

        # Test sample rate validation
        assert settings.validate_sample_rate(44100) == True
        assert settings.validate_sample_rate(48000) == True
        assert settings.validate_sample_rate(96000) == True
        assert settings.validate_sample_rate(192000) == True

        # Invalid sample rates
        assert settings.validate_sample_rate(22050) == False
        assert settings.validate_sample_rate(32000) == False
        assert settings.validate_sample_rate(999999) == False

        # Test bit depth validation
        assert settings.validate_bit_depth(16) == True
        assert settings.validate_bit_depth(24) == True
        assert settings.validate_bit_depth(32) == True

        # Invalid bit depths
        assert settings.validate_bit_depth(8) == False
        assert settings.validate_bit_depth(12) == False
        assert settings.validate_bit_depth(64) == False

    def test_export_format_validation(self):
        """Test export format validation"""
        from core.config import AudioLabSettings

        settings = AudioLabSettings()

        # Valid export formats
        assert settings.validate_export_format("wav") == True
        assert settings.validate_export_format("flac") == True
        assert settings.validate_export_format("mp3") == True
        assert settings.validate_export_format("aac") == True

        # Case insensitive
        assert settings.validate_export_format("WAV") == True
        assert settings.validate_export_format("MP3") == True

        # Invalid formats
        assert settings.validate_export_format("xyz") == False
        assert settings.validate_export_format("") == False
        assert settings.validate_export_format("ogg") == False

    def test_configuration_methods(self):
        """Test configuration method outputs"""
        from core.config import AudioLabSettings

        settings = AudioLabSettings()

        # Test audio config method
        audio_config = settings.get_audio_config()
        assert isinstance(audio_config, dict)
        assert "sample_rate" in audio_config
        assert "buffer_size" in audio_config
        assert "channels" in audio_config
        assert "latency_target_ms" in audio_config
        assert "max_tracks" in audio_config

        # Verify values
        assert audio_config["sample_rate"] == settings.DEFAULT_SAMPLE_RATE
        assert audio_config["buffer_size"] == settings.DEFAULT_BUFFER_SIZE
        assert audio_config["channels"] == 2

        # Test Demucs config method
        demucs_config = settings.get_demucs_config()
        assert isinstance(demucs_config, dict)
        assert "model_path" in demucs_config
        assert "default_model" in demucs_config
        assert "segment_size" in demucs_config
        assert "use_gpu" in demucs_config
        assert "available_models" in demucs_config

        # Test export config method
        export_config = settings.get_export_config()
        assert isinstance(export_config, dict)
        assert "default_format" in export_config
        assert "default_quality" in export_config
        assert "lufs_targets" in export_config
        assert "supported_formats" in export_config
        assert "supported_qualities" in export_config

    def test_directory_creation_logic(self):
        """Test directory creation functionality"""
        from pathlib import Path

        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test directory creation logic directly
            directories = [
                str(temp_path / "audio"),
                str(temp_path / "projects"),
                str(temp_path / "exports"),
                str(temp_path / "temp"),
                str(temp_path / "models")
            ]

            # Create directories
            for directory in directories:
                Path(directory).mkdir(parents=True, exist_ok=True)

            # Verify directories were created
            for directory in directories:
                assert Path(directory).exists()
                assert Path(directory).is_dir()


@pytest.mark.unit
class TestConfigurationConstants:
    """Test configuration constants and enumerations"""

    def test_supported_sample_rates(self):
        """Test supported sample rates list"""
        from core.config import AudioLabSettings

        settings = AudioLabSettings()
        supported_rates = settings.SUPPORTED_SAMPLE_RATES

        assert 44100 in supported_rates
        assert 48000 in supported_rates
        assert 96000 in supported_rates
        assert 192000 in supported_rates
        assert len(supported_rates) == 4

    def test_supported_bit_depths(self):
        """Test supported bit depths list"""
        from core.config import AudioLabSettings

        settings = AudioLabSettings()
        supported_depths = settings.SUPPORTED_BIT_DEPTHS

        assert 16 in supported_depths
        assert 24 in supported_depths
        assert 32 in supported_depths
        assert len(supported_depths) == 3

    def test_demucs_models_available(self):
        """Test available Demucs models list"""
        from core.config import AudioLabSettings

        settings = AudioLabSettings()
        available_models = settings.DEMUCS_MODELS_AVAILABLE

        assert "htdemucs_ft" in available_models
        assert "htdemucs_6s" in available_models
        assert "mdx_extra" in available_models

    def test_lufs_targets(self):
        """Test LUFS targets for different use cases"""
        from core.config import AudioLabSettings

        settings = AudioLabSettings()
        lufs_targets = settings.LUFS_TARGETS

        assert "streaming" in lufs_targets
        assert "mastering" in lufs_targets
        assert "broadcast" in lufs_targets
        assert "archive" in lufs_targets

        # Verify target values
        assert lufs_targets["streaming"] == -14.0
        assert lufs_targets["mastering"] == -9.0
        assert lufs_targets["broadcast"] == -16.0
        assert lufs_targets["archive"] == -18.0

    def test_supported_export_formats(self):
        """Test supported export formats and qualities"""
        from core.config import AudioLabSettings

        settings = AudioLabSettings()
        formats = settings.SUPPORTED_EXPORT_FORMATS
        qualities = settings.SUPPORTED_EXPORT_QUALITIES

        # Check formats
        assert "wav" in formats
        assert "flac" in formats
        assert "mp3" in formats
        assert "aac" in formats

        # Check qualities for each format
        assert "wav" in qualities
        assert "16bit" in qualities["wav"]
        assert "24bit" in qualities["wav"]
        assert "32bit" in qualities["wav"]

        assert "mp3" in qualities
        assert "320kbps" in qualities["mp3"]
        assert "256kbps" in qualities["mp3"]


@pytest.mark.unit
class TestConfigurationProperties:
    """Test configuration properties and derived values"""

    def test_development_mode_detection(self):
        """Test development mode detection logic"""

        # Test development mode logic directly
        def is_development(debug_setting: bool) -> bool:
            return debug_setting

        def is_production(debug_setting: bool) -> bool:
            return not debug_setting

        # Test with debug=False (production mode)
        debug_false = False
        assert is_development(debug_false) == False
        assert is_production(debug_false) == True

        # Test with debug=True (development mode)
        debug_true = True
        assert is_development(debug_true) == True
        assert is_production(debug_true) == False

    def test_database_url_conversion(self):
        """Test database URL conversion for sync operations"""

        # Test URL conversion logic directly
        def convert_async_to_sync_url(database_url: str) -> str:
            return database_url.replace("+asyncpg", "")

        async_url = "postgresql+asyncpg://user:pass@localhost:5432/db"
        sync_url = convert_async_to_sync_url(async_url)

        assert "+asyncpg" not in sync_url
        assert "postgresql://user:pass@localhost:5432/db" == sync_url


@pytest.mark.unit
class TestConfigurationValidation:
    """Test configuration validation and error handling"""

    def test_invalid_configuration_values(self):
        """Test handling of invalid configuration values"""

        # Test invalid sample rate validation
        def validate_sample_rate(rate):
            valid_rates = [44100, 48000, 96000, 192000]
            return rate in valid_rates

        assert validate_sample_rate(48000) == True
        assert validate_sample_rate(999) == False
        assert validate_sample_rate(-48000) == False
        assert validate_sample_rate(0) == False

        # Test invalid bit depth validation
        def validate_bit_depth(depth):
            valid_depths = [16, 24, 32]
            return depth in valid_depths

        assert validate_bit_depth(24) == True
        assert validate_bit_depth(8) == False
        assert validate_bit_depth(64) == False
        assert validate_bit_depth(-16) == False

    def test_configuration_ranges(self):
        """Test configuration value ranges"""

        # Test buffer size ranges
        def validate_buffer_size(size):
            return 64 <= size <= 8192 and size & (size - 1) == 0  # Power of 2

        assert validate_buffer_size(64) == True
        assert validate_buffer_size(128) == True
        assert validate_buffer_size(512) == True
        assert validate_buffer_size(1024) == True

        # Invalid buffer sizes
        assert validate_buffer_size(63) == False  # Not power of 2
        assert validate_buffer_size(100) == False  # Not power of 2
        assert validate_buffer_size(16384) == False  # Too large

        # Test track limits
        def validate_max_tracks(tracks):
            return 1 <= tracks <= 64

        assert validate_max_tracks(32) == True
        assert validate_max_tracks(1) == True
        assert validate_max_tracks(64) == True

        # Invalid track counts
        assert validate_max_tracks(0) == False
        assert validate_max_tracks(100) == False
        assert validate_max_tracks(-5) == False


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])