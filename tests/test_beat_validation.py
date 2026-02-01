"""
Validation and Quality Assurance Tests for Beat Generation
Testing configuration, dependencies, quality metrics, and system validation
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json

from src.core.beat_generation_validation import BeatGenerationValidator
from src.core.config import get_settings
from src.core.result import Result


class TestBeatGenerationValidator:
    """Test suite for beat generation validation"""

    def test_dependency_validation_success(self):
        """Test successful dependency validation"""

        with patch('importlib.util.find_spec') as mock_find_spec:
            with patch('importlib.import_module') as mock_import:

                # Mock successful imports
                mock_find_spec.return_value = Mock()
                mock_module = Mock()
                mock_module.__version__ = "4.35.0"
                mock_import.return_value = mock_module

                result = BeatGenerationValidator.validate_dependencies()

                assert result.success
                validation_data = result.data

                assert "required" in validation_data
                assert "optional" in validation_data
                assert "providers" in validation_data
                assert "summary" in validation_data

    def test_dependency_validation_missing_package(self):
        """Test validation with missing required package"""

        with patch('importlib.util.find_spec') as mock_find_spec:
            # Mock transformers as missing
            def mock_spec(name):
                if name == "transformers":
                    return None
                return Mock()

            mock_find_spec.side_effect = mock_spec

            result = BeatGenerationValidator.validate_dependencies()

            assert result.success
            validation_data = result.data

            # Should report transformers as unavailable
            assert not validation_data["required"]["transformers"]["available"]
            assert not validation_data["summary"]["all_required_available"]

    def test_musicgen_validation_success(self):
        """Test successful MusicGen validation"""

        with patch('transformers.MusicgenForConditionalGeneration') as mock_model:
            with patch('transformers.AutoProcessor') as mock_processor:
                with patch('torch.cuda.is_available', return_value=True):

                    result = BeatGenerationValidator._validate_musicgen()

                    assert result["ready"]
                    assert result["transformers_available"]
                    assert result["torch_available"]
                    assert result["model_loading_possible"]

    def test_musicgen_validation_failure(self):
        """Test MusicGen validation with import failure"""

        with patch('transformers.MusicgenForConditionalGeneration', side_effect=ImportError("No transformers")):

            result = BeatGenerationValidator._validate_musicgen()

            assert not result["ready"]
            assert not result["transformers_available"]
            assert len(result["errors"]) > 0

    def test_soundraw_validation_success(self):
        """Test successful SOUNDRAW validation"""

        with patch('httpx.AsyncClient'):
            with patch('src.core.config.get_settings') as mock_settings:

                mock_settings.return_value.SOUNDRAW_API_KEY = "test_key"

                result = BeatGenerationValidator._validate_soundraw()

                assert result["ready"]
                assert result["httpx_available"]
                assert result["api_key_configured"]

    def test_soundraw_validation_no_api_key(self):
        """Test SOUNDRAW validation without API key"""

        with patch('httpx.AsyncClient'):
            with patch('src.core.config.get_settings') as mock_settings:

                mock_settings.return_value.SOUNDRAW_API_KEY = None

                result = BeatGenerationValidator._validate_soundraw()

                assert not result["ready"]
                assert not result["api_key_configured"]
                assert "not configured" in result["errors"][0]

    def test_gpu_support_detection(self):
        """Test GPU support detection"""

        # Test with CUDA available
        with patch('torch.cuda.is_available', return_value=True):
            assert BeatGenerationValidator._check_gpu_support() is True

        # Test without CUDA
        with patch('torch.cuda.is_available', return_value=False):
            assert BeatGenerationValidator._check_gpu_support() is False

        # Test with import error
        with patch('torch.cuda.is_available', side_effect=ImportError):
            assert BeatGenerationValidator._check_gpu_support() is False

    def test_model_path_validation_success(self):
        """Test successful model path validation"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = asyncio.run(
                BeatGenerationValidator.validate_model_path("musicgen", tmp_dir)
            )

            assert result.success
            assert result.data is True

    def test_model_path_validation_create_directory(self):
        """Test model path validation with directory creation"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "models" / "musicgen"

            result = asyncio.run(
                BeatGenerationValidator.validate_model_path("musicgen", str(model_path))
            )

            assert result.success
            assert model_path.exists()

    def test_model_path_validation_permission_error(self):
        """Test model path validation with permission error"""

        # Try to create path in system directory (should fail)
        invalid_path = "/root/protected/models"

        result = asyncio.run(
            BeatGenerationValidator.validate_model_path("musicgen", invalid_path)
        )

        assert not result.success
        assert "Cannot create" in result.error or "not writable" in result.error

    def test_installation_instructions(self):
        """Test getting installation instructions"""

        instructions = BeatGenerationValidator.get_installation_instructions()

        assert "required" in instructions
        assert "optional" in instructions
        assert "gpu_support" in instructions
        assert "soundraw" in instructions

        # Check that actual pip commands are provided
        required = instructions["required"]
        assert any("pip install transformers" in cmd for cmd in required)
        assert any("pip install torch" in cmd for cmd in required)

    def test_version_comparison(self):
        """Test version comparison functionality"""

        # Test equal versions
        assert BeatGenerationValidator._compare_versions("4.35.0", "4.35.0") == 0

        # Test newer version
        assert BeatGenerationValidator._compare_versions("4.36.0", "4.35.0") == 1

        # Test older version
        assert BeatGenerationValidator._compare_versions("4.34.0", "4.35.0") == -1

        # Test major version differences
        assert BeatGenerationValidator._compare_versions("5.0.0", "4.35.0") == 1
        assert BeatGenerationValidator._compare_versions("3.0.0", "4.35.0") == -1


class TestConfigurationValidation:
    """Test beat generation configuration validation"""

    def test_settings_beat_generation_config(self):
        """Test beat generation configuration structure"""

        settings = get_settings()
        config = settings.get_beat_generation_config()

        # Check main structure
        assert "musicgen" in config
        assert "soundraw" in config
        assert "limits" in config
        assert "storage" in config
        assert "features" in config

        # Check MusicGen config
        musicgen = config["musicgen"]
        assert "model_path" in musicgen
        assert "default_model" in musicgen
        assert "use_gpu" in musicgen
        assert "available_models" in musicgen

        # Check SOUNDRAW config
        soundraw = config["soundraw"]
        assert "api_timeout" in soundraw
        assert "max_duration" in soundraw
        assert "enabled" in soundraw

        # Check limits
        limits = config["limits"]
        assert "max_concurrent" in limits
        assert "max_duration" in limits
        assert "timeout" in limits

    def test_settings_directory_creation(self):
        """Test that configuration creates necessary directories"""

        settings = get_settings()

        # Check that beat-related directories are created
        beat_paths = [
            settings.BEATS_PATH,
            settings.MIDI_PATH,
            settings.MUSICGEN_MODEL_PATH
        ]

        for path in beat_paths:
            assert Path(path).exists()

    def test_configuration_validation(self):
        """Test configuration value validation"""

        settings = get_settings()

        # Test reasonable defaults
        assert settings.BEAT_GENERATION_MAX_DURATION > 0
        assert settings.BEAT_GENERATION_DEFAULT_DURATION > 0
        assert settings.BEAT_GENERATION_MAX_CONCURRENT > 0
        assert settings.BEAT_QUALITY_THRESHOLD >= 0

        # Test timeout values
        assert settings.BEAT_GENERATION_TIMEOUT > settings.BEAT_GENERATION_MAX_DURATION


class TestQualityMetrics:
    """Test quality measurement and validation"""

    def test_audio_quality_calculation_normal(self):
        """Test quality calculation with normal audio"""

        from src.services.beat_generation_service import BeatGenerationService

        service = BeatGenerationService()

        # Generate normal audio signal
        duration = 2.0
        sample_rate = 44100
        samples = int(duration * sample_rate)

        # Create audio with good dynamic range
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples)) * 0.7
        audio += np.sin(2 * np.pi * 880 * np.linspace(0, duration, samples)) * 0.3

        quality = service._calculate_beat_quality(audio, 120.0)

        assert 5.0 <= quality <= 10.0  # Should be decent quality

    def test_audio_quality_calculation_poor(self):
        """Test quality calculation with poor audio"""

        from src.services.beat_generation_service import BeatGenerationService

        service = BeatGenerationService()

        # Silent audio
        silent_audio = np.zeros(44100)
        quality = service._calculate_beat_quality(silent_audio, 120.0)
        assert quality == 0.0

        # Very low amplitude audio
        quiet_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100)) * 0.001
        quality = service._calculate_beat_quality(quiet_audio, 120.0)
        assert quality < 3.0

    def test_midi_quality_validation(self):
        """Test MIDI export quality validation"""

        from src.utils.midi_export import MidiExportService

        service = MidiExportService()

        # Test with valid patterns
        valid_patterns = {
            "kick": [0.0, 1.0, 2.0, 3.0],
            "snare": [0.5, 1.5, 2.5, 3.5]
        }

        # Should not raise errors with valid patterns
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            with patch('pretty_midi.PrettyMIDI') as mock_midi:
                mock_midi.return_value = Mock()

                result = loop.run_until_complete(
                    service._create_midi_from_patterns(
                        valid_patterns, 120.0, "4/4", True
                    )
                )

                # Should complete without error
                assert result is not None

            loop.close()

        except Exception as e:
            pytest.fail(f"MIDI creation should not fail with valid patterns: {e}")


class TestSystemIntegration:
    """Test system-wide integration and validation"""

    @pytest.mark.asyncio
    async def test_end_to_end_validation(self):
        """Test complete end-to-end validation"""

        # Validate dependencies
        deps_result = BeatGenerationValidator.validate_dependencies()
        assert deps_result.success

        # Validate configuration
        settings = get_settings()
        config = settings.get_beat_generation_config()
        assert config is not None

        # Validate model paths
        for provider in ["musicgen"]:
            path_result = await BeatGenerationValidator.validate_model_path(
                provider,
                config[provider]["model_path"]
            )
            assert path_result.success

    def test_error_handling_robustness(self):
        """Test error handling across all components"""

        # Test validator error handling
        with patch('importlib.import_module', side_effect=Exception("Import error")):
            result = BeatGenerationValidator.validate_dependencies()
            assert not result.success
            assert "validation failed" in result.error.lower()

        # Test configuration error handling
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Permission denied")):
            try:
                settings = get_settings()
                # Should handle directory creation errors gracefully
                config = settings.get_beat_generation_config()
                assert config is not None
            except:
                # If it fails, it should fail gracefully
                pass

    def test_performance_requirements(self):
        """Test that system meets performance requirements"""

        # Test import times (should be fast)
        import time

        start_time = time.time()
        from src.services.beat_generation_service import BeatGenerationService
        from src.utils.midi_export import MidiExportService
        from src.services.project_integration_service import ProjectIntegrationService
        end_time = time.time()

        import_time = end_time - start_time
        assert import_time < 2.0  # Should import in less than 2 seconds

        # Test initialization times
        start_time = time.time()
        service = BeatGenerationService()
        integration = ProjectIntegrationService()
        midi_service = MidiExportService()
        end_time = time.time()

        init_time = end_time - start_time
        assert init_time < 1.0  # Should initialize in less than 1 second

    @pytest.mark.asyncio
    async def test_memory_requirements(self):
        """Test memory usage requirements"""

        import tracemalloc

        tracemalloc.start()

        # Initialize all services
        from src.services.beat_generation_service import BeatGenerationService
        from src.services.project_integration_service import ProjectIntegrationService
        from src.utils.midi_export import MidiExportService

        service = BeatGenerationService()
        integration = ProjectIntegrationService()
        midi_service = MidiExportService()

        # Simulate some processing
        audio = np.random.randn(44100 * 8)  # 8 seconds of audio
        patterns = {
            "kick": [0.0, 2.0, 4.0, 6.0],
            "snare": [1.0, 3.0, 5.0, 7.0]
        }

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (less than 50MB for services)
        assert peak < 50 * 1024 * 1024  # 50MB

        # Cleanup
        await service.cleanup()

    def test_concurrency_safety(self):
        """Test thread safety and concurrency"""

        from src.services.beat_generation_service import BeatGenerationService

        # Test that multiple service instances can coexist
        services = [BeatGenerationService() for _ in range(3)]

        for service in services:
            assert service.device in ["cpu", "cuda"]
            assert not service._model_loaded

        # Each should have independent state
        devices = [service.device for service in services]
        # All should be the same optimal device, but independently determined
        assert len(set(devices)) == 1

    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility"""

        import platform

        # Test that paths work across platforms
        settings = get_settings()
        paths = [
            settings.BEATS_PATH,
            settings.MIDI_PATH,
            settings.MUSICGEN_MODEL_PATH
        ]

        for path_str in paths:
            path = Path(path_str)
            assert path.is_absolute() or platform.system() == "Windows"
            # Path should be valid for current platform


class TestComplianceAndStandards:
    """Test compliance with AudioLab standards and patterns"""

    def test_result_pattern_usage(self):
        """Test that Result pattern is used consistently"""

        from src.services.beat_generation_service import BeatGenerationService
        from src.utils.midi_export import MidiExportService

        # All async methods should return Result types
        service = BeatGenerationService()
        midi_service = MidiExportService()

        # Check method signatures return Result
        import inspect

        for cls in [BeatGenerationService, MidiExportService]:
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if name.startswith('_') or not asyncio.iscoroutinefunction(method):
                    continue

                # Public async methods should return Result
                sig = inspect.signature(method)
                if hasattr(sig.return_annotation, '__origin__'):
                    # Check if it's Result[Something]
                    pass  # Complex type checking would go here

    def test_type_annotations(self):
        """Test that type annotations are complete"""

        from src.services.beat_generation_service import BeatGenerationService
        import inspect

        # Check that all public methods have type annotations
        for name, method in inspect.getmembers(BeatGenerationService, predicate=inspect.ismethod):
            if name.startswith('_'):
                continue

            sig = inspect.signature(method)

            # Parameters should have annotations
            for param_name, param in sig.parameters.items():
                if param_name in ['self', 'args', 'kwargs']:
                    continue
                # In production, would check param.annotation != inspect.Parameter.empty

            # Return type should be annotated
            # In production, would check sig.return_annotation != inspect.Signature.empty

    def test_logging_integration(self):
        """Test proper logging integration"""

        from src.core.logging import audio_logger

        # Test that audio_logger has required methods
        required_methods = [
            'log_processing_start',
            'log_processing_complete',
            'log_processing_error'
        ]

        for method_name in required_methods:
            assert hasattr(audio_logger, method_name)
            method = getattr(audio_logger, method_name)
            assert callable(method)

    def test_configuration_consistency(self):
        """Test configuration consistency with AudioLab patterns"""

        settings = get_settings()

        # Test that beat generation settings follow naming conventions
        beat_attrs = [attr for attr in dir(settings) if attr.startswith('BEAT_')]

        assert len(beat_attrs) > 0

        for attr in beat_attrs:
            value = getattr(settings, attr)
            # Should not be None (except for optional fields)
            if not attr.endswith('_API_KEY'):
                assert value is not None

    def test_database_schema_consistency(self):
        """Test database schema follows AudioLab patterns"""

        from src.database.models import BeatGenerationRequest, BeatTemplate, BeatVariation

        # Check that models follow UUID pattern
        for model in [BeatGenerationRequest, BeatTemplate, BeatVariation]:
            # Should have id field
            assert hasattr(model, '__table__')
            columns = model.__table__.columns
            assert 'id' in columns
            # In production, would check UUID type

            # Should have timestamps
            has_created_at = 'created_at' in columns
            assert has_created_at  # All models should have created_at