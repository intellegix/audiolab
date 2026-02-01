"""
Integration tests for Beat Generation and Project Integration
Testing the complete workflow from generation to project placement
"""

import pytest
import asyncio
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import uuid
from decimal import Decimal

from src.services.project_integration_service import ProjectIntegrationService
from src.services.beat_workflow_manager import BeatWorkflowManager
from src.utils.midi_export import MidiExportService
from src.core.result import Result
from src.database.schemas import BeatGenerationRequest


class TestProjectIntegrationService:
    """Test suite for ProjectIntegrationService"""

    @pytest.fixture
    async def integration_service(self):
        """Create integration service instance"""
        return ProjectIntegrationService()

    @pytest.fixture
    def sample_audio_file(self):
        """Create temporary audio file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            # Create minimal WAV file header
            tmp.write(b'RIFF')
            tmp.write((36).to_bytes(4, 'little'))  # File size - 8
            tmp.write(b'WAVE')
            tmp.write(b'fmt ')
            tmp.write((16).to_bytes(4, 'little'))  # PCM header size
            tmp.write((1).to_bytes(2, 'little'))   # Audio format (PCM)
            tmp.write((1).to_bytes(2, 'little'))   # Channels
            tmp.write((44100).to_bytes(4, 'little')) # Sample rate
            tmp.write((88200).to_bytes(4, 'little')) # Byte rate
            tmp.write((2).to_bytes(2, 'little'))   # Block align
            tmp.write((16).to_bytes(2, 'little'))  # Bits per sample
            tmp.write(b'data')
            tmp.write((0).to_bytes(4, 'little'))   # Data size

            yield tmp.name

        Path(tmp.name).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_add_beat_to_track(self, integration_service, sample_audio_file):
        """Test adding beat to track"""

        track_id = uuid.uuid4()
        timeline_position = 16.0
        clip_name = "Generated Beat"

        result = await integration_service.add_beat_to_track(
            sample_audio_file,
            track_id,
            timeline_position,
            clip_name
        )

        assert result.success
        data = result.data
        assert data["track_id"] == track_id
        assert data["timeline_position"] == timeline_position
        assert data["clip_name"] == clip_name
        assert "clip_id" in data

    @pytest.mark.asyncio
    async def test_add_beat_nonexistent_file(self, integration_service):
        """Test adding non-existent audio file"""

        result = await integration_service.add_beat_to_track(
            "/nonexistent/file.wav",
            uuid.uuid4(),
            0.0,
            "Test Beat"
        )

        assert not result.success
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_add_beat_with_midi_export(self, integration_service, sample_audio_file):
        """Test adding beat with automatic MIDI export"""

        with patch.object(integration_service.midi_service, 'convert_beat_to_midi') as mock_midi:
            mock_midi.return_value = Result.ok(Mock(data="/path/to/beat.mid"))

            result = await integration_service.add_beat_with_midi_export(
                sample_audio_file,
                uuid.uuid4(),
                0.0,
                120.0,
                "4/4",
                "Beat with MIDI",
                auto_export_midi=True
            )

            assert result.success
            data = result.data
            assert data["midi_exported"] is True
            assert data["tempo"] == 120.0
            assert data["time_signature"] == "4/4"

    @pytest.mark.asyncio
    async def test_create_beat_arrangement(self, integration_service):
        """Test creating beat arrangement from variations"""

        project_id = uuid.uuid4()
        beat_variations = [
            {"name": "verse", "audio_path": "/path/to/verse.wav"},
            {"name": "chorus", "audio_path": "/path/to/chorus.wav"},
            {"name": "bridge", "audio_path": "/path/to/bridge.wav"}
        ]
        arrangement_pattern = ["verse", "verse", "chorus", "verse", "chorus", "bridge", "chorus"]

        with patch.object(integration_service, 'create_beat_track_template') as mock_track:
            with patch.object(integration_service, 'add_beat_with_midi_export') as mock_add:

                mock_track.return_value = Result.ok(uuid.uuid4())
                mock_add.return_value = Result.ok({"clip_id": uuid.uuid4()})

                result = await integration_service.create_beat_arrangement(
                    project_id,
                    beat_variations,
                    arrangement_pattern,
                    120.0,
                    "4/4"
                )

                assert result.success
                data = result.data
                assert len(data["clip_ids"]) == len(arrangement_pattern)
                assert data["arrangement_pattern"] == arrangement_pattern
                assert data["tempo"] == 120.0

    @pytest.mark.asyncio
    async def test_sync_beat_to_project_grid(self, integration_service, sample_audio_file):
        """Test beat synchronization to project grid"""

        result = await integration_service.sync_beat_to_project_grid(
            sample_audio_file,
            130.0,  # Target tempo
            "4/4",
            "1/16"  # Grid resolution
        )

        assert result.success
        sync_path = result.data
        assert Path(sync_path).exists()
        assert "130bpm" in sync_path

    @pytest.mark.asyncio
    async def test_auto_organize_beats_by_style(self, integration_service):
        """Test automatic beat organization by style"""

        project_id = uuid.uuid4()
        beat_files = [
            {"path": "/path/to/rock1.wav", "style_tags": {"genre": "rock"}},
            {"path": "/path/to/rock2.wav", "style_tags": {"genre": "rock"}},
            {"path": "/path/to/jazz1.wav", "style_tags": {"genre": "jazz"}},
            {"path": "/path/to/hip_hop1.wav", "style_tags": {"genre": "hip-hop"}}
        ]

        with patch.object(integration_service, 'create_beat_track_template') as mock_track:
            with patch.object(integration_service, 'add_beat_to_track') as mock_add:

                # Mock track creation
                track_ids = {
                    "rock": uuid.uuid4(),
                    "jazz": uuid.uuid4(),
                    "hip-hop": uuid.uuid4()
                }
                mock_track.side_effect = lambda _, name: Result.ok(
                    track_ids[name.split(" - ")[1].lower()]
                )

                # Mock clip addition
                mock_add.return_value = Result.ok({"clip_id": uuid.uuid4()})

                result = await integration_service.auto_organize_beats_by_style(
                    project_id,
                    beat_files
                )

                assert result.success
                organization = result.data

                assert "rock" in organization
                assert "jazz" in organization
                assert "hip-hop" in organization
                assert len(organization["rock"]) == 2  # Two rock beats
                assert len(organization["jazz"]) == 1   # One jazz beat

    def test_get_recommended_loop_bars(self, integration_service):
        """Test recommended loop bar calculation"""

        # Slow tempo
        slow_bars = integration_service.get_recommended_loop_bars(70.0)
        assert slow_bars == 8

        # Normal tempo
        normal_bars = integration_service.get_recommended_loop_bars(120.0)
        assert normal_bars == 4

        # Fast tempo
        fast_bars = integration_service.get_recommended_loop_bars(160.0)
        assert fast_bars == 2


class TestMidiExportService:
    """Test suite for MIDI export functionality"""

    @pytest.fixture
    def midi_service(self):
        """Create MIDI export service instance"""
        return MidiExportService()

    @pytest.fixture
    def sample_beat_audio(self):
        """Generate sample beat audio for MIDI export testing"""

        # 4 seconds of audio at 44.1kHz
        duration = 4.0
        sample_rate = 44100
        samples = int(duration * sample_rate)

        # Create a simple beat pattern
        audio = np.zeros(samples)

        # Add kick drums at beats 1 and 3
        kick_freq = 60  # 60 Hz for kick
        kick_duration = 0.1  # 100ms
        kick_samples = int(kick_duration * sample_rate)

        beat_interval = sample_rate // 2  # 120 BPM = 2 beats per second

        for beat in [0, 2]:  # Beats 1 and 3
            start = beat * beat_interval
            end = start + kick_samples
            if end < samples:
                kick_wave = np.sin(2 * np.pi * kick_freq * np.linspace(0, kick_duration, kick_samples))
                audio[start:end] += kick_wave * 0.5

        # Add snare on beats 2 and 4
        snare_freq = 200  # 200 Hz for snare
        for beat in [1, 3]:  # Beats 2 and 4
            start = beat * beat_interval
            end = start + kick_samples
            if end < samples:
                snare_wave = np.sin(2 * np.pi * snare_freq * np.linspace(0, kick_duration, kick_samples))
                audio[start:end] += snare_wave * 0.3

        return audio

    @pytest.mark.asyncio
    async def test_onset_detection(self, midi_service, sample_beat_audio):
        """Test onset detection in sample beat"""

        onsets = await midi_service._detect_onsets(sample_beat_audio, sensitivity=0.3)

        # Should detect onsets near beat positions
        assert len(onsets) >= 2  # At least kick and snare
        assert onsets[0] < 0.5   # First onset near beginning

    @pytest.mark.asyncio
    async def test_drum_pattern_analysis(self, midi_service, sample_beat_audio):
        """Test drum pattern analysis"""

        # Mock onset times at beat positions
        onsets = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])

        patterns = await midi_service._analyze_drum_patterns(sample_beat_audio, onsets)

        assert "kick" in patterns
        assert "snare" in patterns
        assert "hihat" in patterns

        # Should have some onsets in each pattern
        total_onsets = sum(len(pattern) for pattern in patterns.values())
        assert total_onsets == len(onsets)

    @pytest.mark.skipif(
        not pytest.importorskip("pretty_midi", minversion="0.2.10"),
        reason="pretty_midi not available"
    )
    @pytest.mark.asyncio
    async def test_midi_creation_from_patterns(self, midi_service):
        """Test MIDI file creation from drum patterns"""

        patterns = {
            "kick": [0.0, 1.0, 2.0, 3.0],
            "snare": [0.5, 1.5, 2.5, 3.5],
            "hihat": [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]
        }

        midi_data = await midi_service._create_midi_from_patterns(
            patterns,
            tempo=120.0,
            time_signature="4/4",
            quantize=True
        )

        assert midi_data.initial_tempo == 120.0
        assert len(midi_data.instruments) == 1

        drum_track = midi_data.instruments[0]
        assert drum_track.is_drum

        # Check that notes were created
        assert len(drum_track.notes) > 0

        # Check for correct MIDI note numbers
        note_numbers = [note.pitch for note in drum_track.notes]
        assert midi_service.DRUM_MAP["kick"] in note_numbers
        assert midi_service.DRUM_MAP["snare"] in note_numbers

    @pytest.mark.asyncio
    async def test_midi_template_creation(self, midi_service):
        """Test MIDI template creation"""

        with patch('src.utils.midi_export.MIDI_AVAILABLE', True):
            with patch('pretty_midi.PrettyMIDI') as mock_midi:
                with patch('pretty_midi.Instrument') as mock_instrument:
                    with patch('pretty_midi.Note') as mock_note:

                        mock_midi_instance = Mock()
                        mock_midi.return_value = mock_midi_instance
                        mock_midi_instance.write = Mock()

                        result = await midi_service.create_midi_template(
                            tempo=120.0,
                            time_signature="4/4",
                            pattern_type="basic_rock",
                            bars=4
                        )

                        assert result.success
                        assert result.data.endswith('.mid')

    def test_get_supported_patterns(self, midi_service):
        """Test getting supported pattern templates"""

        patterns = midi_service.get_supported_patterns()

        assert len(patterns) > 0
        assert "basic_rock" in patterns
        assert "basic_pop" in patterns
        assert "hip_hop" in patterns
        assert "jazz" in patterns

    def test_validate_midi_dependencies(self, midi_service):
        """Test MIDI dependency validation"""

        deps = midi_service.validate_midi_dependencies()

        assert "pretty_midi" in deps
        assert "mido" in deps
        assert "librosa" in deps

        # All values should be boolean
        for available in deps.values():
            assert isinstance(available, bool)


class TestBeatWorkflowManager:
    """Test suite for complete beat workflow management"""

    @pytest.fixture
    async def workflow_manager(self):
        """Create workflow manager instance"""
        return BeatWorkflowManager()

    @pytest.fixture
    def sample_beat_request(self):
        """Create sample beat generation request"""
        return BeatGenerationRequest(
            project_id=uuid.uuid4(),
            prompt="energetic rock drum beat",
            provider="musicgen",
            model_name="facebook/musicgen-small",
            duration=Decimal("8.0"),
            tempo=Decimal("120.0"),
            time_signature="4/4",
            style_tags={"genre": "rock", "energy": "high"}
        )

    @pytest.mark.asyncio
    async def test_complete_workflow_execution(self, workflow_manager, sample_beat_request):
        """Test complete workflow from generation to integration"""

        project_id = uuid.uuid4()
        user_id = uuid.uuid4()

        progress_updates = []
        def progress_callback(progress: float, stage: str):
            progress_updates.append((progress, stage))

        with patch.object(workflow_manager.generation_service, 'generate_beat') as mock_generate:
            with patch.object(workflow_manager, '_save_generated_beats') as mock_save:
                with patch.object(workflow_manager, '_export_midi_for_beats') as mock_midi:
                    with patch.object(workflow_manager, '_integrate_beats_into_project') as mock_integrate:

                        # Mock successful generation
                        mock_audio = np.random.randn(44100 * 8)
                        mock_generate.return_value = ProcessingResult(
                            success=True,
                            data=mock_audio,
                            metadata={"quality_score": 8.5, "provider": "musicgen"}
                        )

                        # Mock file saving
                        mock_save.return_value = Result.ok([{
                            "type": "primary",
                            "audio_path": "/path/to/beat.wav",
                            "metadata": {"quality_score": 8.5}
                        }])

                        # Mock MIDI export
                        mock_midi.return_value = {"/path/to/beat.wav": "/path/to/beat.mid"}

                        # Mock project integration
                        mock_integrate.return_value = Result.ok({
                            "track_ids": [uuid.uuid4()],
                            "clip_ids": [uuid.uuid4()],
                            "primary_clip_id": uuid.uuid4()
                        })

                        result = await workflow_manager.execute_complete_beat_workflow(
                            sample_beat_request,
                            project_id,
                            user_id,
                            auto_add_to_project=True,
                            generate_variations=1,
                            progress_callback=progress_callback
                        )

                        assert result.success
                        workflow_data = result.data

                        # Verify workflow structure
                        assert "workflow_id" in workflow_data
                        assert "primary_beat" in workflow_data
                        assert "variations" in workflow_data
                        assert "project_integration" in workflow_data
                        assert "workflow_stats" in workflow_data

                        # Verify progress callbacks were called
                        assert len(progress_updates) > 0
                        assert any("Initializing" in stage for _, stage in progress_updates)
                        assert any("completed" in stage for _, stage in progress_updates)

    @pytest.mark.asyncio
    async def test_beat_variation_generation(self, workflow_manager, sample_beat_request):
        """Test beat variation generation"""

        with patch.object(workflow_manager.generation_service, 'generate_beat') as mock_generate:

            # Mock successful generations with different parameters
            mock_generate.return_value = ProcessingResult(
                success=True,
                data=np.random.randn(44100 * 8),
                metadata={"quality_score": 7.8}
            )

            variations = await workflow_manager._generate_beat_variations(
                sample_beat_request,
                variation_count=3
            )

            assert len(variations) == 3

            # Verify each variation has required fields
            for variation in variations:
                assert "strategy" in variation
                assert "audio_data" in variation
                assert "metadata" in variation
                assert "variation_parameters" in variation

            # Verify different strategies were used
            strategies = [var["strategy"] for var in variations]
            assert len(set(strategies)) > 1  # Different strategies

    @pytest.mark.asyncio
    async def test_beat_library_creation(self, workflow_manager):
        """Test creating beat library from template"""

        project_id = uuid.uuid4()

        with patch.object(workflow_manager, 'execute_complete_beat_workflow') as mock_workflow:

            # Mock successful workflow executions
            mock_workflow.return_value = Result.ok({
                "primary_beat": {
                    "audio_path": "/path/to/beat.wav",
                    "quality_score": 8.0
                },
                "variations": [],
                "workflow_stats": {"total_beats_generated": 1}
            })

            result = await workflow_manager.create_beat_library_from_template(
                project_id,
                template_style="rock",
                tempo_range=(100, 140),
                variation_count=3
            )

            assert result.success
            library_data = result.data

            assert library_data["library_style"] == "rock"
            assert library_data["beats_generated"] == 3
            assert library_data["tempo_range"] == (100, 140)
            assert len(library_data["beat_details"]) == 3

            # Verify different tempos were used
            tempos = [beat["tempo"] for beat in library_data["beat_details"]]
            assert min(tempos) >= 100
            assert max(tempos) <= 140
            assert len(set(tempos)) > 1  # Different tempos

    def test_tempo_categorization(self, workflow_manager):
        """Test tempo categorization for organization"""

        assert workflow_manager._get_tempo_category(70.0) == "slow"
        assert workflow_manager._get_tempo_category(90.0) == "medium_slow"
        assert workflow_manager._get_tempo_category(110.0) == "medium"
        assert workflow_manager._get_tempo_category(130.0) == "medium_fast"
        assert workflow_manager._get_tempo_category(160.0) == "fast"


class TestPerformanceAndScalability:
    """Performance and scalability tests"""

    @pytest.mark.asyncio
    async def test_concurrent_beat_generation(self):
        """Test concurrent beat generation requests"""

        async def generate_beat(prompt: str):
            service = BeatGenerationService()
            with patch.object(service, 'generate_beat') as mock_generate:
                mock_generate.return_value = ProcessingResult(
                    success=True,
                    data=np.random.randn(44100),
                    metadata={"quality_score": 8.0}
                )
                return await service.generate_beat(prompt, 1.0, 120.0, "4/4")

        # Generate multiple beats concurrently
        prompts = ["rock beat", "jazz beat", "electronic beat", "hip hop beat"]

        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*[generate_beat(prompt) for prompt in prompts])
        end_time = asyncio.get_event_loop().time()

        # All should succeed
        assert all(result.success for result in results)

        # Should complete in reasonable time (less than if run sequentially)
        assert end_time - start_time < 5.0

    @pytest.mark.asyncio
    async def test_memory_usage_during_batch_processing(self):
        """Test memory usage during batch beat processing"""

        import tracemalloc

        tracemalloc.start()

        workflow_manager = BeatWorkflowManager()

        # Process multiple beats
        for i in range(5):
            request = BeatGenerationRequest(
                project_id=uuid.uuid4(),
                prompt=f"beat {i}",
                provider="musicgen",
                duration=Decimal("2.0"),
                tempo=Decimal("120.0"),
                time_signature="4/4"
            )

            with patch.object(workflow_manager.generation_service, 'generate_beat') as mock_generate:
                mock_generate.return_value = ProcessingResult(
                    success=True,
                    data=np.random.randn(44100 * 2),
                    metadata={"quality_score": 8.0}
                )

                await workflow_manager.execute_complete_beat_workflow(
                    request,
                    uuid.uuid4(),
                    uuid.uuid4(),
                    auto_add_to_project=False
                )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Memory usage should be reasonable (less than 100MB for this test)
        assert peak < 100 * 1024 * 1024  # 100MB

    @pytest.mark.asyncio
    async def test_large_beat_library_creation(self):
        """Test creating large beat library"""

        workflow_manager = BeatWorkflowManager()

        with patch.object(workflow_manager, 'execute_complete_beat_workflow') as mock_workflow:
            mock_workflow.return_value = Result.ok({
                "primary_beat": {"audio_path": "/path/to/beat.wav"},
                "variations": [],
                "workflow_stats": {"total_beats_generated": 1}
            })

            start_time = asyncio.get_event_loop().time()

            result = await workflow_manager.create_beat_library_from_template(
                uuid.uuid4(),
                "electronic",
                tempo_range=(80, 180),
                variation_count=20  # Large library
            )

            end_time = asyncio.get_event_loop().time()

            assert result.success
            assert result.data["beats_generated"] == 20

            # Should complete in reasonable time
            assert end_time - start_time < 30.0  # Less than 30 seconds for 20 beats


class TestErrorRecoveryAndResilience:
    """Test error recovery and system resilience"""

    @pytest.mark.asyncio
    async def test_workflow_partial_failure_recovery(self):
        """Test recovery from partial workflow failures"""

        workflow_manager = BeatWorkflowManager()

        request = BeatGenerationRequest(
            project_id=uuid.uuid4(),
            prompt="test beat",
            provider="musicgen",
            duration=Decimal("4.0"),
            tempo=Decimal("120.0"),
            time_signature="4/4"
        )

        with patch.object(workflow_manager.generation_service, 'generate_beat') as mock_generate:
            with patch.object(workflow_manager, '_export_midi_for_beats') as mock_midi:
                with patch.object(workflow_manager, '_integrate_beats_into_project') as mock_integrate:

                    # Generation succeeds
                    mock_generate.return_value = ProcessingResult(
                        success=True,
                        data=np.random.randn(44100 * 4),
                        metadata={"quality_score": 8.0}
                    )

                    # MIDI export fails
                    mock_midi.return_value = None

                    # Integration fails
                    mock_integrate.return_value = Result.err("Integration failed")

                    result = await workflow_manager.execute_complete_beat_workflow(
                        request,
                        uuid.uuid4(),
                        uuid.uuid4(),
                        auto_add_to_project=True
                    )

                    # Workflow should still succeed even with partial failures
                    assert result.success
                    workflow_data = result.data

                    # Primary beat should be available
                    assert "primary_beat" in workflow_data
                    assert workflow_data["primary_beat"]["audio_path"]

                    # Integration should show failure
                    assert workflow_data["project_integration"] is None

    @pytest.mark.asyncio
    async def test_service_cleanup_on_error(self):
        """Test that services are properly cleaned up on errors"""

        service = BeatGenerationService()

        # Mock an error during generation
        with patch.object(service, 'generate_beat', side_effect=Exception("Generation failed")):
            try:
                await service.generate_beat("test", 1.0, 120.0, "4/4")
            except:
                pass

        # Service should still be cleanable
        await service.cleanup()

        assert not service._model_loaded
        assert service.current_model is None