"""
Test suite for Beat Generation API endpoints
Testing REST API and WebSocket integration
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from httpx import AsyncClient
from fastapi.testclient import TestClient
import uuid
from decimal import Decimal

from src.api.routes.audio import router
from src.database.schemas import BeatGenerationRequest, BeatGenerationJobResponse
from src.core.result import Result


@pytest.fixture
def client():
    """Create test client"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/audio")
    return TestClient(app)


@pytest.fixture
def async_client():
    """Create async test client"""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/audio")
    return AsyncClient(app=app, base_url="http://test")


@pytest.fixture
def mock_beat_request():
    """Mock beat generation request"""
    return {
        "project_id": str(uuid.uuid4()),
        "prompt": "energetic rock drum beat",
        "provider": "musicgen",
        "model_name": "facebook/musicgen-small",
        "duration": 8.0,
        "tempo": 120.0,
        "time_signature": "4/4",
        "style_tags": {
            "genre": "rock",
            "energy": "high",
            "mood": "aggressive"
        }
    }


class TestBeatGenerationAPI:
    """Test suite for beat generation API endpoints"""

    @pytest.mark.asyncio
    async def test_generate_beat_endpoint(self, async_client, mock_beat_request):
        """Test /beats/generate endpoint"""

        with patch('src.services.beat_generation_service.BeatGenerationService') as mock_service:
            with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:

                # Mock repository response
                mock_db_request = Mock()
                mock_db_request.id = uuid.uuid4()
                mock_repo.return_value.create_beat_generation_request.return_value = Result.ok(mock_db_request)

                response = await async_client.post(
                    "/api/v1/audio/beats/generate",
                    json=mock_beat_request
                )

                assert response.status_code == 200
                data = response.json()

                assert "request_id" in data
                assert "estimated_duration" in data
                assert "status" in data
                assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_generate_beat_invalid_duration(self, async_client, mock_beat_request):
        """Test beat generation with invalid duration"""

        # Set duration too high
        mock_beat_request["duration"] = 100.0

        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value.get_beat_generation_config.return_value = {
                "limits": {"max_duration": 30.0}
            }

            response = await async_client.post(
                "/api/v1/audio/beats/generate",
                json=mock_beat_request
            )

            assert response.status_code == 400
            assert "exceeds maximum" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_generate_beat_soundraw_not_configured(self, async_client, mock_beat_request):
        """Test SOUNDRAW generation without API key"""

        mock_beat_request["provider"] = "soundraw"

        with patch('src.core.config.get_settings') as mock_settings:
            mock_settings.return_value.get_beat_generation_config.return_value = {
                "soundraw": {"enabled": False},
                "limits": {"max_duration": 30.0}
            }

            response = await async_client.post(
                "/api/v1/audio/beats/generate",
                json=mock_beat_request
            )

            assert response.status_code == 400
            assert "SOUNDRAW provider not configured" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_beat_generation_status(self, async_client):
        """Test /beats/{request_id}/status endpoint"""

        request_id = uuid.uuid4()

        with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
            # Mock completed beat request
            mock_beat_request = Mock()
            mock_beat_request.id = request_id
            mock_beat_request.project_id = uuid.uuid4()
            mock_beat_request.user_id = uuid.uuid4()
            mock_beat_request.prompt = "test beat"
            mock_beat_request.provider = "musicgen"
            mock_beat_request.model_name = "facebook/musicgen-small"
            mock_beat_request.duration = Decimal("8.0")
            mock_beat_request.tempo = Decimal("120.0")
            mock_beat_request.time_signature = "4/4"
            mock_beat_request.style_tags = {}
            mock_beat_request.status = "completed"
            mock_beat_request.progress = Decimal("100.0")
            mock_beat_request.current_stage = None
            mock_beat_request.generated_audio_path = "/path/to/beat.wav"
            mock_beat_request.generated_midi_path = "/path/to/beat.mid"
            mock_beat_request.quality_score = Decimal("8.5")
            mock_beat_request.processing_time = Decimal("15.3")
            mock_beat_request.provider_metadata = {"model": "musicgen-small"}
            mock_beat_request.error_message = None
            mock_beat_request.created_at = "2024-01-01T12:00:00"
            mock_beat_request.started_at = "2024-01-01T12:00:05"
            mock_beat_request.completed_at = "2024-01-01T12:00:20"

            mock_repo.return_value.get_beat_generation_request.return_value = Result.ok(mock_beat_request)

            response = await async_client.get(f"/api/v1/audio/beats/{request_id}/status")

            assert response.status_code == 200
            data = response.json()

            assert data["id"] == str(request_id)
            assert data["status"] == "completed"
            assert data["progress"] == 100.0
            assert data["generated_audio_path"] == "/path/to/beat.wav"
            assert data["quality_score"] == 8.5

    @pytest.mark.asyncio
    async def test_get_beat_status_not_found(self, async_client):
        """Test status endpoint with non-existent request"""

        request_id = uuid.uuid4()

        with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
            mock_repo.return_value.get_beat_generation_request.return_value = Result.ok(None)

            response = await async_client.get(f"/api/v1/audio/beats/{request_id}/status")

            assert response.status_code == 404
            assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_add_beat_to_project(self, async_client):
        """Test /beats/{request_id}/add-to-project endpoint"""

        request_id = uuid.uuid4()
        track_id = uuid.uuid4()

        request_data = {
            "track_id": str(track_id),
            "timeline_position": 0.0,
            "clip_name": "Generated Beat"
        }

        with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
            with patch('src.services.project_integration_service.ProjectIntegrationService') as mock_integration:

                # Mock beat request
                mock_beat_request = Mock()
                mock_beat_request.status = "completed"
                mock_beat_request.generated_audio_path = "/path/to/beat.wav"
                mock_beat_request.prompt = "test beat"
                mock_repo.return_value.get_beat_generation_request.return_value = Result.ok(mock_beat_request)

                # Mock integration service
                mock_integration.return_value.add_beat_to_track.return_value = Result.ok({
                    "clip_id": uuid.uuid4()
                })

                response = await async_client.post(
                    f"/api/v1/audio/beats/{request_id}/add-to-project",
                    json=request_data
                )

                assert response.status_code == 200
                data = response.json()

                assert data["success"] is True
                assert "clip_id" in data
                assert data["message"] == "Beat added to project successfully"

    @pytest.mark.asyncio
    async def test_add_beat_to_project_not_completed(self, async_client):
        """Test adding incomplete beat to project"""

        request_id = uuid.uuid4()
        track_id = uuid.uuid4()

        request_data = {
            "track_id": str(track_id),
            "timeline_position": 0.0
        }

        with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
            # Mock pending beat request
            mock_beat_request = Mock()
            mock_beat_request.status = "processing"
            mock_beat_request.generated_audio_path = None
            mock_repo.return_value.get_beat_generation_request.return_value = Result.ok(mock_beat_request)

            response = await async_client.post(
                f"/api/v1/audio/beats/{request_id}/add-to-project",
                json=request_data
            )

            assert response.status_code == 400
            assert "not completed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_beat_templates(self, async_client):
        """Test /beats/templates endpoint"""

        with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
            # Mock templates
            mock_template = Mock()
            mock_template.id = uuid.uuid4()
            mock_template.name = "Rock Beat"
            mock_template.description = "Standard rock drum pattern"
            mock_template.category = "rock"
            mock_template.tags = ["rock", "4/4", "medium"]
            mock_template.default_tempo = Decimal("120.0")
            mock_template.time_signature = "4/4"
            mock_template.duration = Decimal("8.0")
            mock_template.provider_config = {"provider": "musicgen"}
            mock_template.prompt_template = "rock drum beat"
            mock_template.is_public = True
            mock_template.usage_count = 42
            mock_template.average_quality = Decimal("8.2")
            mock_template.created_by_user_id = None
            mock_template.created_at = "2024-01-01T12:00:00"
            mock_template.updated_at = "2024-01-01T12:00:00"

            mock_repo.return_value.get_beat_templates.return_value = Result.ok([mock_template])

            response = await async_client.get("/api/v1/audio/beats/templates")

            assert response.status_code == 200
            data = response.json()

            assert len(data) == 1
            template = data[0]
            assert template["name"] == "Rock Beat"
            assert template["category"] == "rock"
            assert template["usage_count"] == 42

    @pytest.mark.asyncio
    async def test_get_beat_templates_with_filters(self, async_client):
        """Test beat templates with category and search filters"""

        with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
            mock_repo.return_value.get_beat_templates.return_value = Result.ok([])

            response = await async_client.get(
                "/api/v1/audio/beats/templates",
                params={"category": "hip-hop", "search": "trap,808"}
            )

            assert response.status_code == 200

            # Verify repository was called with correct filters
            call_args = mock_repo.return_value.get_beat_templates.call_args
            assert call_args[1]["category"] == "hip-hop"
            assert call_args[1]["search_tags"] == ["trap", "808"]

    @pytest.mark.asyncio
    async def test_get_beat_variations(self, async_client):
        """Test /beats/{request_id}/variations endpoint"""

        request_id = uuid.uuid4()

        with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
            # Mock variations
            mock_variation = Mock()
            mock_variation.id = uuid.uuid4()
            mock_variation.beat_generation_request_id = request_id
            mock_variation.variation_index = 1
            mock_variation.name = "Variation 1"
            mock_variation.audio_path = "/path/to/variation1.wav"
            mock_variation.midi_path = "/path/to/variation1.mid"
            mock_variation.quality_score = Decimal("8.0")
            mock_variation.user_rating = 4
            mock_variation.generation_seed = 12345
            mock_variation.generation_metadata = {"tempo_mod": 1.1}
            mock_variation.is_selected = True
            mock_variation.used_in_project = False
            mock_variation.created_at = "2024-01-01T12:00:00"

            mock_repo.return_value.get_request_variations.return_value = Result.ok([mock_variation])

            response = await async_client.get(f"/api/v1/audio/beats/{request_id}/variations")

            assert response.status_code == 200
            data = response.json()

            assert len(data) == 1
            variation = data[0]
            assert variation["variation_index"] == 1
            assert variation["is_selected"] is True
            assert variation["quality_score"] == 8.0

    @pytest.mark.asyncio
    async def test_select_beat_variation(self, async_client):
        """Test /beats/variations/{variation_id}/select endpoint"""

        variation_id = uuid.uuid4()

        with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
            mock_variation = Mock()
            mock_repo.return_value.select_variation.return_value = Result.ok(mock_variation)

            response = await async_client.post(f"/api/v1/audio/beats/variations/{variation_id}/select")

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["variation_id"] == str(variation_id)

    @pytest.mark.asyncio
    async def test_get_beat_providers(self, async_client):
        """Test /beats/providers endpoint"""

        with patch('src.core.config.get_settings') as mock_settings:
            with patch('src.services.beat_generation_service.BeatGenerationService') as mock_service:

                # Mock configuration
                mock_settings.return_value.get_beat_generation_config.return_value = {
                    "musicgen": {"max_duration": 30.0},
                    "soundraw": {"enabled": True, "max_duration": 300.0},
                    "limits": {"max_duration": 30.0, "max_concurrent": 2}
                }

                # Mock service
                mock_service.return_value.get_supported_models.return_value = [
                    "facebook/musicgen-small",
                    "facebook/musicgen-medium"
                ]

                response = await async_client.get("/api/v1/audio/beats/providers")

                assert response.status_code == 200
                data = response.json()

                assert "providers" in data
                assert "musicgen" in data["providers"]
                assert "soundraw" in data["providers"]
                assert data["default_provider"] == "musicgen"

                musicgen = data["providers"]["musicgen"]
                assert musicgen["local"] is True
                assert musicgen["max_duration"] == 30.0
                assert len(musicgen["models"]) > 0

    def test_estimate_generation_duration(self):
        """Test generation duration estimation"""

        from src.api.routes.audio import _estimate_generation_duration

        # Test MusicGen estimation
        musicgen_duration = _estimate_generation_duration("musicgen", 8.0)
        assert musicgen_duration > 8  # Should be more than beat duration
        assert musicgen_duration < 60  # But reasonable

        # Test SOUNDRAW estimation
        soundraw_duration = _estimate_generation_duration("soundraw", 8.0)
        assert soundraw_duration == 60  # Fixed duration for API

        # Test unknown provider
        unknown_duration = _estimate_generation_duration("unknown", 8.0)
        assert unknown_duration == 120  # Default


class TestBeatGenerationWorkflow:
    """Integration tests for complete beat generation workflow"""

    @pytest.mark.asyncio
    async def test_complete_workflow_success(self, async_client, mock_beat_request):
        """Test complete successful workflow from generation to project integration"""

        request_id = uuid.uuid4()

        with patch('src.services.beat_generation_service.BeatGenerationService') as mock_service:
            with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
                with patch('src.services.project_integration_service.ProjectIntegrationService') as mock_integration:

                    # Mock database creation
                    mock_db_request = Mock()
                    mock_db_request.id = request_id
                    mock_repo.return_value.create_beat_generation_request.return_value = Result.ok(mock_db_request)

                    # Start generation
                    response = await async_client.post(
                        "/api/v1/audio/beats/generate",
                        json=mock_beat_request
                    )

                    assert response.status_code == 200
                    job_data = response.json()
                    assert job_data["request_id"] == str(request_id)

                    # Mock completion status
                    mock_db_request.status = "completed"
                    mock_db_request.generated_audio_path = "/path/to/beat.wav"
                    mock_db_request.quality_score = Decimal("8.5")
                    mock_repo.return_value.get_beat_generation_request.return_value = Result.ok(mock_db_request)

                    # Check status
                    status_response = await async_client.get(f"/api/v1/audio/beats/{request_id}/status")
                    assert status_response.status_code == 200
                    status_data = status_response.json()
                    assert status_data["status"] == "completed"

                    # Add to project
                    mock_integration.return_value.add_beat_to_track.return_value = Result.ok({
                        "clip_id": uuid.uuid4()
                    })

                    project_response = await async_client.post(
                        f"/api/v1/audio/beats/{request_id}/add-to-project",
                        json={
                            "track_id": str(uuid.uuid4()),
                            "timeline_position": 0.0
                        }
                    )

                    assert project_response.status_code == 200
                    project_data = project_response.json()
                    assert project_data["success"] is True

    @pytest.mark.asyncio
    async def test_workflow_with_error_handling(self, async_client, mock_beat_request):
        """Test workflow error handling"""

        with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
            # Mock database error
            mock_repo.return_value.create_beat_generation_request.return_value = Result.err("Database error")

            response = await async_client.post(
                "/api/v1/audio/beats/generate",
                json=mock_beat_request
            )

            assert response.status_code == 400
            assert "Database error" in response.json()["detail"]


class TestWebSocketIntegration:
    """Test WebSocket integration for real-time progress updates"""

    @pytest.mark.asyncio
    async def test_websocket_progress_updates(self):
        """Test WebSocket progress update integration"""

        from src.api.websocket import ConnectionManager
        from src.database.schemas import BeatGenerationProgressEvent

        manager = ConnectionManager()

        # Mock WebSocket connection
        mock_websocket = Mock()
        mock_websocket.send_text = AsyncMock()

        connection_id = "test_connection"
        await manager.connect(mock_websocket, connection_id, "test_project")

        # Test progress event
        progress_event = BeatGenerationProgressEvent(
            request_id=uuid.uuid4(),
            status="processing",
            progress=Decimal("50.0"),
            current_stage="Generating beat audio"
        )

        await manager.send_beat_generation_progress(connection_id, progress_event)

        # Verify WebSocket message was sent
        mock_websocket.send_text.assert_called_once()
        sent_message = json.loads(mock_websocket.send_text.call_args[0][0])

        assert sent_message["type"] == "beat_generation_progress"
        assert sent_message["data"]["progress"] == 50.0
        assert sent_message["data"]["status"] == "processing"

    def test_beat_generation_callback_creation(self):
        """Test creation of beat generation progress callback"""

        from src.api.websocket import ConnectionManager

        manager = ConnectionManager()
        callback = manager.create_beat_generation_callback("test_conn", "test_request_id")

        assert callable(callback)
        # Callback function created successfully


class TestValidationAndErrorCases:
    """Test validation and error handling"""

    @pytest.mark.asyncio
    async def test_invalid_request_validation(self, async_client):
        """Test request validation"""

        # Missing required fields
        invalid_request = {
            "prompt": "test beat"
            # Missing project_id, duration, etc.
        }

        response = await async_client.post(
            "/api/v1/audio/beats/generate",
            json=invalid_request
        )

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_invalid_uuid_handling(self, async_client):
        """Test handling of invalid UUIDs"""

        response = await async_client.get("/api/v1/audio/beats/invalid-uuid/status")
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_database_connection_error(self, async_client, mock_beat_request):
        """Test handling of database connection errors"""

        with patch('src.database.repositories.beat_generation_repository.BeatGenerationRepository') as mock_repo:
            mock_repo.side_effect = Exception("Database connection failed")

            response = await async_client.post(
                "/api/v1/audio/beats/generate",
                json=mock_beat_request
            )

            assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_service_initialization_error(self, async_client, mock_beat_request):
        """Test handling of service initialization errors"""

        with patch('src.services.beat_generation_service.BeatGenerationService') as mock_service:
            mock_service.side_effect = Exception("Service initialization failed")

            response = await async_client.post(
                "/api/v1/audio/beats/generate",
                json=mock_beat_request
            )

            assert response.status_code == 500