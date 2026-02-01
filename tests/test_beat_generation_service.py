"""
Test suite for Beat Generation Service
Comprehensive testing of MusicGen and SOUNDRAW integration
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal
import tempfile
from pathlib import Path

from src.services.beat_generation_service import BeatGenerationService
from src.core.result import Result
from src.core.audio_processor import ProcessingResult


class TestBeatGenerationService:
    """Test suite for BeatGenerationService"""

    @pytest.fixture
    async def service(self):
        """Create service instance for testing"""
        service = BeatGenerationService()
        yield service
        await service.cleanup()

    @pytest.fixture
    def sample_audio(self):
        """Generate sample audio data for testing"""
        # 8 seconds of 44kHz mono audio
        duration = 8.0
        sample_rate = 44100
        samples = int(duration * sample_rate)

        # Generate a simple beat pattern (kick and snare)
        audio = np.zeros(samples)
        kick_samples = np.sin(2 * np.pi * 60 * np.linspace(0, 0.1, int(0.1 * sample_rate)))
        snare_samples = np.sin(2 * np.pi * 200 * np.linspace(0, 0.1, int(0.1 * sample_rate)))

        # Add kicks and snares at regular intervals
        beat_interval = int(sample_rate * 0.5)  # Every 0.5 seconds
        for i in range(0, samples, beat_interval):
            if i + len(kick_samples) < samples:
                audio[i:i + len(kick_samples)] += kick_samples
            if i + beat_interval//2 + len(snare_samples) < samples:
                audio[i + beat_interval//2:i + beat_interval//2 + len(snare_samples)] += snare_samples

        return audio

    def test_service_initialization(self):
        """Test service initialization and device detection"""
        service = BeatGenerationService()

        assert service.device in ["cpu", "cuda"]
        assert service.max_memory_gb > 0
        assert not service._model_loaded
        assert service.current_provider is None

    @pytest.mark.asyncio
    async def test_musicgen_model_loading(self, service):
        """Test MusicGen model loading"""

        with patch('src.services.beat_generation_service.MUSICGEN_AVAILABLE', True):
            with patch('transformers.MusicgenForConditionalGeneration.from_pretrained') as mock_model:
                with patch('transformers.AutoProcessor.from_pretrained') as mock_processor:

                    mock_model.return_value = Mock()
                    mock_processor.return_value = Mock()

                    result = await service.load_model("musicgen", "facebook/musicgen-small")

                    assert result.success
                    assert service.current_provider == "musicgen"
                    assert service._model_loaded

    @pytest.mark.asyncio
    async def test_musicgen_model_loading_failure(self, service):
        """Test MusicGen model loading failure"""

        with patch('src.services.beat_generation_service.MUSICGEN_AVAILABLE', False):
            result = await service.load_model("musicgen", "facebook/musicgen-small")

            assert not result.success
            assert "MusicGen not available" in result.error

    @pytest.mark.asyncio
    async def test_soundraw_client_initialization(self, service):
        """Test SOUNDRAW client initialization"""

        service.soundraw_api_key = "test_key"

        with patch('src.services.beat_generation_service.HTTPX_AVAILABLE', True):
            with patch('httpx.AsyncClient') as mock_client:
                mock_instance = AsyncMock()
                mock_instance.get.return_value.status_code = 200
                mock_client.return_value = mock_instance

                result = await service.load_model("soundraw")

                assert result.success
                assert service.current_provider == "soundraw"

    @pytest.mark.asyncio
    async def test_beat_generation_without_model(self, service):
        """Test beat generation without loaded model"""

        result = await service.generate_beat(
            prompt="hip hop beat",
            duration=8.0,
            tempo=120.0,
            time_signature="4/4"
        )

        assert not result.success
        assert "Model not loaded" in result.error

    @pytest.mark.asyncio
    async def test_musicgen_beat_generation(self, service):
        """Test beat generation with MusicGen"""

        with patch('src.services.beat_generation_service.MUSICGEN_AVAILABLE', True):
            # Mock model and processor
            mock_model = Mock()
            mock_processor = Mock()

            # Mock generation result
            mock_audio = np.random.randn(44100 * 8)  # 8 seconds of audio
            mock_model.generate.return_value = Mock()
            mock_model.generate.return_value[0].cpu.return_value.numpy.return_value = mock_audio
            mock_model.config.audio_encoder.sampling_rate = 44100
            mock_model.config.audio_encoder.hop_length = 512

            service._model = mock_model
            service._processor = mock_processor
            service.current_model = "facebook/musicgen-small"
            service.current_provider = "musicgen"
            service._model_loaded = True

            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=mock_audio)

                result = await service.generate_beat(
                    prompt="energetic rock beat",
                    duration=8.0,
                    tempo=120.0,
                    time_signature="4/4",
                    style_tags={"genre": "rock", "energy": "high"}
                )

                assert result.success
                assert result.data is not None
                assert "quality_score" in result.metadata
                assert result.metadata["provider"] == "musicgen"

    @pytest.mark.asyncio
    async def test_beat_generation_duration_limit(self, service):
        """Test beat generation duration limit enforcement"""

        # Mock model as loaded
        service._model_loaded = True
        service.current_provider = "musicgen"

        result = await service.generate_beat(
            prompt="test beat",
            duration=100.0,  # Exceeds 30s limit for MusicGen
            tempo=120.0,
            time_signature="4/4"
        )

        assert not result.success
        assert "exceeds maximum" in result.error

    @pytest.mark.asyncio
    async def test_gpu_memory_fallback(self, service):
        """Test GPU memory exhaustion fallback to CPU"""

        with patch('src.services.beat_generation_service.MUSICGEN_AVAILABLE', True):
            with patch('torch.cuda.OutOfMemoryError', Exception):

                # Mock model loading that fails on GPU
                service.device = "cuda"

                with patch('transformers.MusicgenForConditionalGeneration.from_pretrained') as mock_model:
                    mock_model.side_effect = Exception("CUDA out of memory")

                    result = await service.load_model("musicgen", "facebook/musicgen-small")

                    # Should attempt CPU fallback
                    assert service.device == "cpu"

    def test_prompt_enhancement(self, service):
        """Test prompt enhancement with musical context"""

        enhanced = service._enhance_musicgen_prompt(
            "drum beat",
            120.0,
            "4/4",
            {"genre": "rock", "mood": "energetic"}
        )

        assert "120 BPM" in enhanced
        assert "4/4 time" in enhanced
        assert "rock style" in enhanced
        assert "energetic mood" in enhanced

    @pytest.mark.asyncio
    async def test_tempo_synchronization(self, service):
        """Test tempo synchronization functionality"""

        # Mock audio data
        audio = np.random.randn(44100 * 8)

        result = await service._synchronize_tempo(
            audio,
            44100,
            130.0,  # Target tempo
            "4/4"
        )

        # For now, should return original audio (placeholder implementation)
        assert np.array_equal(result, audio)

    def test_quality_calculation(self, service):
        """Test beat quality calculation"""

        # Test with normal audio
        normal_audio = np.random.randn(44100) * 0.5
        quality = service._calculate_beat_quality(normal_audio, 120.0)
        assert 0.0 <= quality <= 10.0

        # Test with silent audio
        silent_audio = np.zeros(44100)
        quality = service._calculate_beat_quality(silent_audio, 120.0)
        assert quality == 0.0

        # Test with clipped audio
        clipped_audio = np.ones(44100)
        quality = service._calculate_beat_quality(clipped_audio, 120.0)
        assert quality > 0.0

    def test_supported_models(self, service):
        """Test getting supported models for providers"""

        musicgen_models = service.get_supported_models("musicgen")
        assert len(musicgen_models) > 0
        assert "facebook/musicgen-small" in musicgen_models

        soundraw_models = service.get_supported_models("soundraw")
        assert len(soundraw_models) == 0  # SOUNDRAW doesn't expose model names

        unknown_models = service.get_supported_models("unknown")
        assert len(unknown_models) == 0

    def test_max_duration(self, service):
        """Test getting maximum duration for providers"""

        musicgen_max = service.get_max_duration("musicgen")
        assert musicgen_max == 30.0

        soundraw_max = service.get_max_duration("soundraw")
        assert soundraw_max == 300.0

        unknown_max = service.get_max_duration("unknown")
        assert unknown_max == 30.0  # Default

    @pytest.mark.asyncio
    async def test_progress_callback(self, service):
        """Test progress callback functionality"""

        progress_updates = []

        def progress_callback(progress: float, stage: str):
            progress_updates.append((progress, stage))

        service.set_progress_callback(progress_callback)

        # Mock model loading to trigger progress callbacks
        with patch('src.services.beat_generation_service.MUSICGEN_AVAILABLE', True):
            with patch('transformers.MusicgenForConditionalGeneration.from_pretrained') as mock_model:
                with patch('transformers.AutoProcessor.from_pretrained') as mock_processor:

                    mock_model.return_value = Mock()
                    mock_processor.return_value = Mock()

                    await service.load_model("musicgen", "facebook/musicgen-small")

                    # Should have received progress updates
                    assert len(progress_updates) > 0
                    assert any("Loading" in stage for _, stage in progress_updates)

    @pytest.mark.asyncio
    async def test_service_cleanup(self, service):
        """Test service cleanup functionality"""

        # Set up some mock state
        service._model = Mock()
        service._processor = Mock()
        service.current_model = "test"
        service._model_loaded = True

        await service.cleanup()

        # Should clean up all state
        assert service._model is None
        assert service._processor is None
        assert not service._model_loaded
        assert service.current_model is None

    @pytest.mark.asyncio
    async def test_concurrent_generation_prevention(self, service):
        """Test that concurrent generation is prevented"""

        # Mock model as loaded
        service._model_loaded = True
        service.current_provider = "musicgen"

        # Mock a long-running generation
        async def slow_generation(*args, **kwargs):
            await asyncio.sleep(0.1)
            return ProcessingResult(success=True, data=np.zeros(1000))

        with patch.object(service, '_generate_musicgen_beat', side_effect=slow_generation):

            # Start first generation
            task1 = asyncio.create_task(service.generate_beat(
                prompt="test", duration=1.0, tempo=120.0, time_signature="4/4"
            ))

            # Try to start second generation immediately
            result2 = await service.generate_beat(
                prompt="test", duration=1.0, tempo=120.0, time_signature="4/4"
            )

            # Second should be rejected
            assert not result2.success
            assert "already running" in result2.error.lower()

            # Wait for first to complete
            await task1

    def test_device_selection_logic(self):
        """Test device selection logic"""

        # Test with CUDA available
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value.total_memory = 8 * 1024**3  # 8GB

                service = BeatGenerationService()
                assert service.device == "cuda"

        # Test with insufficient GPU memory
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.get_device_properties') as mock_props:
                mock_props.return_value.total_memory = 1 * 1024**3  # 1GB

                service = BeatGenerationService()
                assert service.device == "cpu"

        # Test with no CUDA
        with patch('torch.cuda.is_available', return_value=False):
            service = BeatGenerationService()
            assert service.device == "cpu"


@pytest.mark.integration
class TestBeatGenerationIntegration:
    """Integration tests for beat generation with real dependencies"""

    @pytest.mark.skipif(
        not pytest.importorskip("transformers", minversion="4.35.0"),
        reason="MusicGen dependencies not available"
    )
    @pytest.mark.asyncio
    async def test_real_musicgen_small_model(self):
        """Test with real MusicGen small model (requires internet)"""

        service = BeatGenerationService()

        try:
            # Load small model (fastest for testing)
            result = await service.load_model("musicgen", "facebook/musicgen-small")

            if result.success:
                # Generate a short beat
                generation_result = await service.generate_beat(
                    prompt="simple drum beat",
                    duration=2.0,  # Very short for testing
                    tempo=120.0,
                    time_signature="4/4"
                )

                assert generation_result.success
                assert generation_result.data is not None
                assert len(generation_result.data) > 0

        finally:
            await service.cleanup()

    @pytest.mark.asyncio
    async def test_error_handling_robustness(self):
        """Test error handling in various failure scenarios"""

        service = BeatGenerationService()

        # Test with invalid provider
        result = await service.load_model("invalid_provider")
        assert not result.success

        # Test with invalid model name
        result = await service.load_model("musicgen", "invalid/model")
        assert not result.success

        # Test generation without model
        result = await service.generate_beat("test", 1.0, 120.0, "4/4")
        assert not result.success