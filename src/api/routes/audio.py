"""
AudioLab Audio Processing API Routes
REST endpoints for audio processing operations with real Demucs integration
Includes real-time recording and playback capabilities for overdubbing
"""

from typing import List, Optional
import uuid
import asyncio
from decimal import Decimal
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from ...core.audio_processor import AudioFileManager, ProcessingResult
from ...services.audio_processing_service import audio_processing_service
from ...services.recording_service import get_recording_service
from ...services.playback_service import get_playback_service

router = APIRouter()


class AudioProcessingRequest(BaseModel):
    """Request model for audio processing"""
    file_path: str
    operation: str
    parameters: dict = {}


class AudioProcessingResponse(BaseModel):
    """Response model for audio processing"""
    success: bool
    message: str
    result_path: Optional[str] = None
    metadata: Optional[dict] = None


class StemSeparationRequest(BaseModel):
    """Request model for stem separation"""
    clip_id: uuid.UUID
    model_name: str = "htdemucs_ft"
    force_reprocess: bool = False


class StemSeparationResponse(BaseModel):
    """Response model for stem separation"""
    success: bool
    clip_id: str
    separation_id: Optional[str] = None
    stems: Optional[dict] = None
    model_used: Optional[str] = None
    processing_time: Optional[float] = None
    quality_score: Optional[float] = None
    cached: Optional[bool] = False
    error: Optional[str] = None


class BatchSeparationRequest(BaseModel):
    """Request model for batch stem separation"""
    clip_ids: List[uuid.UUID]
    model_name: str = "htdemucs_ft"


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    name: str
    stems: List[str]
    description: str
    loaded: bool


# Recording and Playback Models

class AudioInputDeviceResponse(BaseModel):
    """Response model for audio input device"""
    device_id: str
    name: str
    channels: int
    sample_rate: int
    max_input_channels: int
    default_sample_rate: float
    is_default: bool
    host_api: str


class StartRecordingRequest(BaseModel):
    """Request model for starting recording"""
    device_id: str
    start_time: float = 0.0


class RecordingResponse(BaseModel):
    """Response model for recording operations"""
    success: bool
    session_id: Optional[str] = None
    track_id: Optional[str] = None
    clip_id: Optional[str] = None
    message: Optional[str] = None
    errors: List[str] = []


class PlaybackRequest(BaseModel):
    """Request model for playback operations"""
    position: float = 0.0


class PlaybackResponse(BaseModel):
    """Response model for playback operations"""
    success: bool
    status: Optional[str] = None
    position: Optional[float] = None
    duration: Optional[float] = None
    message: Optional[str] = None
    errors: List[str] = []


class PlaybackStatusResponse(BaseModel):
    """Response model for playback status"""
    status: str
    position: float
    duration: float
    tempo: Optional[float] = None
    project_id: Optional[str] = None
    project_loaded: bool
    tracks_loaded: int
    sample_rate: int


class TrackRecordingRequest(BaseModel):
    """Request model for track recording configuration"""
    device_id: str
    enabled: bool = True


@router.post("/upload", response_model=dict)
async def upload_audio_file(
    file: UploadFile = File(...)
):
    """Upload audio file for processing"""
    try:
        # Validate file type
        allowed_extensions = ['.wav', '.flac', '.mp3', '.aiff']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {allowed_extensions}"
            )

        # Save uploaded file
        file_path = f"./temp/{file.filename}"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Load and validate audio
        result = await AudioFileManager.load_audio(file_path)

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        return {
            "success": True,
            "message": "Audio file uploaded successfully",
            "file_path": file_path,
            "metadata": result.metadata
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/formats", response_model=List[str])
async def get_supported_formats():
    """Get list of supported audio formats"""
    return list(AudioFileManager.SUPPORTED_FORMATS.values())


@router.post("/process", response_model=AudioProcessingResponse)
async def process_audio(request: AudioProcessingRequest):
    """Process audio file with specified operation"""
    try:
        # Placeholder for audio processing
        # This will be expanded with actual processing logic

        return AudioProcessingResponse(
            success=True,
            message=f"Audio processing '{request.operation}' completed",
            result_path=None,
            metadata={"operation": request.operation}
        )

    except Exception as e:
        return AudioProcessingResponse(
            success=False,
            message=str(e)
        )


@router.post("/separate", response_model=StemSeparationResponse)
async def separate_audio_clip(
    request: StemSeparationRequest,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for progress updates")
):
    """
    Separate audio clip into stems using Demucs AI

    Supports real-time progress tracking via WebSocket if connection_id is provided.
    Results are cached - reprocessing only occurs if force_reprocess is True.
    """
    try:
        result = await audio_processing_service.separate_audio_clip(
            clip_id=request.clip_id,
            model_name=request.model_name,
            connection_id=connection_id,
            force_reprocess=request.force_reprocess
        )

        if result["success"]:
            return StemSeparationResponse(**result)
        else:
            raise HTTPException(status_code=400, detail=result.get("error", "Separation failed"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/separate/batch", response_model=dict)
async def batch_separate_clips(
    request: BatchSeparationRequest,
    background_tasks: BackgroundTasks,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for progress updates")
):
    """
    Separate multiple audio clips in batch with combined progress tracking

    Processing runs in background for large batches to avoid timeout.
    Progress updates sent via WebSocket if connection_id provided.
    """
    try:
        if len(request.clip_ids) > 10:
            # Run large batches in background
            background_tasks.add_task(
                audio_processing_service.batch_separate_clips,
                request.clip_ids,
                request.model_name,
                connection_id
            )
            return {
                "success": True,
                "message": f"Batch processing started for {len(request.clip_ids)} clips",
                "background_processing": True
            }
        else:
            # Process small batches immediately
            result = await audio_processing_service.batch_separate_clips(
                request.clip_ids,
                request.model_name,
                connection_id
            )
            return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[ModelInfoResponse])
async def get_available_models():
    """Get list of available Demucs models and their capabilities"""
    try:
        models = await audio_processing_service.get_available_models()
        return [ModelInfoResponse(**model) for model in models]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/{model_name}/load")
async def preload_model(
    model_name: str,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for progress updates")
):
    """
    Preload a Demucs model into memory for faster processing

    This endpoint allows warming up models before processing to reduce
    first-request latency. Progress updates sent via WebSocket if provided.
    """
    try:
        # Setup progress callback if connection provided
        progress_callback = None
        if connection_id:
            from ...api.websocket import websocket_manager
            progress_callback = websocket_manager.create_progress_callback(
                connection_id,
                "model_loading"
            )

        # Load the model
        await audio_processing_service._ensure_model_loaded(model_name, progress_callback)

        return {
            "success": True,
            "message": f"Model {model_name} loaded successfully",
            "model_name": model_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/separation/{clip_id}/history")
async def get_separation_history(clip_id: uuid.UUID):
    """Get history of stem separations for a clip"""
    try:
        from ...database.connection import get_async_session
        from ...database.repositories.stem_separation_repository import StemSeparationRepository

        async with get_async_session() as session:
            repo = StemSeparationRepository(session)
            separations = await repo.get_by_clip(clip_id)

            history = []
            for sep in separations:
                history.append({
                    "id": str(sep.id),
                    "model_used": sep.model_used,
                    "stems": sep.stems,
                    "processing_time": float(sep.processing_time or 0),
                    "quality_score": float(sep.quality_score or 0),
                    "created_at": sep.created_at.isoformat()
                })

            return {
                "success": True,
                "clip_id": str(clip_id),
                "separation_count": len(history),
                "separations": history
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RECORDING AND PLAYBACK ENDPOINTS
# Real-time recording and overdubbing functionality
# ============================================================================

@router.get("/devices", response_model=List[AudioInputDeviceResponse])
async def get_audio_input_devices():
    """
    List available audio input devices for recording

    Returns all available audio input devices that can be used for recording.
    Includes device capabilities, sample rates, and channel information.
    """
    try:
        recording_service = await get_recording_service()
        devices = await recording_service.get_available_audio_devices()

        return [
            AudioInputDeviceResponse(
                device_id=device.device_id,
                name=device.name,
                channels=device.channels,
                sample_rate=device.sample_rate,
                max_input_channels=device.max_input_channels,
                default_sample_rate=device.default_sample_rate,
                is_default=device.is_default,
                host_api=device.host_api
            )
            for device in devices
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get audio devices: {str(e)}")


@router.post("/tracks/{track_id}/record/enable", response_model=RecordingResponse)
async def enable_track_recording(
    track_id: uuid.UUID,
    request: TrackRecordingRequest
):
    """
    Enable recording on a track with specified input device

    Configures a track for recording by setting the input device and enabling
    the record flag. Required before starting recording sessions.
    """
    try:
        recording_service = await get_recording_service()

        if request.enabled:
            result = await recording_service.enable_track_recording(track_id, request.device_id)
        else:
            result = await recording_service.disable_track_recording(track_id)

        if result["success"]:
            return RecordingResponse(
                success=True,
                track_id=str(track_id),
                message=f"Track recording {'enabled' if request.enabled else 'disabled'}"
            )
        else:
            return RecordingResponse(
                success=False,
                errors=result.get("errors", ["Unknown error"])
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tracks/{track_id}/record/start", response_model=RecordingResponse)
async def start_track_recording(
    track_id: uuid.UUID,
    request: StartRecordingRequest,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for progress updates")
):
    """
    Start recording on a track with specified input device

    Begins real-time audio recording on the specified track. The track must be
    record-enabled with a valid input device. Supports real-time progress
    updates via WebSocket for monitoring recording status.
    """
    try:
        recording_service = await get_recording_service()

        result = await recording_service.start_recording(
            track_id=track_id,
            device_id=request.device_id,
            start_time=Decimal(str(request.start_time)),
            connection_id=connection_id
        )

        if result["success"]:
            return RecordingResponse(
                success=True,
                session_id=result["session_id"],
                track_id=str(track_id),
                message="Recording started successfully"
            )
        else:
            return RecordingResponse(
                success=False,
                errors=result.get("errors", ["Failed to start recording"])
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tracks/{track_id}/record/stop/{session_id}", response_model=RecordingResponse)
async def stop_track_recording(
    track_id: uuid.UUID,
    session_id: uuid.UUID,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for updates")
):
    """
    Stop recording and save audio clip

    Stops the specified recording session and saves the recorded audio as a new
    clip on the track. The clip will be positioned at the recording start time
    with the recorded duration.
    """
    try:
        recording_service = await get_recording_service()

        # Stop and save recording in one operation
        result = await recording_service.stop_and_save_recording(
            session_id=session_id,
            connection_id=connection_id
        )

        if result["success"]:
            return RecordingResponse(
                success=True,
                session_id=str(session_id),
                track_id=str(track_id),
                clip_id=result.get("clip_id"),
                message="Recording stopped and saved as clip"
            )
        else:
            return RecordingResponse(
                success=False,
                errors=result.get("errors", ["Failed to stop recording"])
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recording/status/{session_id}")
async def get_recording_status(session_id: uuid.UUID):
    """Get current status of recording session"""
    try:
        recording_service = await get_recording_service()
        status = await recording_service.get_recording_status(session_id)

        if status:
            return {"success": True, "status": status}
        else:
            return {"success": False, "error": "Recording session not found"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recording/active")
async def get_active_recordings():
    """Get all active recording sessions"""
    try:
        recording_service = await get_recording_service()
        sessions = await recording_service.get_all_active_recordings()

        return {
            "success": True,
            "active_sessions": sessions,
            "count": len(sessions)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PLAYBACK ENDPOINTS
# Multi-track synchronized playback for overdubbing
# ============================================================================

@router.post("/projects/{project_id}/playback/load")
async def load_project_for_playback(project_id: uuid.UUID):
    """
    Load project data for playback

    Pre-loads all tracks and clips for the specified project into the playback
    engine. Required before starting playback operations.
    """
    try:
        playback_service = await get_playback_service()
        result = await playback_service.load_project(project_id)

        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("errors", ["Failed to load project"]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/playback/play", response_model=PlaybackResponse)
async def start_playback(
    project_id: uuid.UUID,
    request: PlaybackRequest,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for position updates")
):
    """
    Start project playback from specified position

    Begins synchronized playback of all tracks in the project. Supports starting
    from any timeline position for overdubbing workflows. Real-time position
    updates sent via WebSocket if connection_id provided.
    """
    try:
        playback_service = await get_playback_service()

        # Ensure project is loaded first
        load_result = await playback_service.load_project(project_id)
        if not load_result["success"]:
            return PlaybackResponse(
                success=False,
                errors=load_result.get("errors", ["Failed to load project"])
            )

        # Start playback
        result = await playback_service.play(
            start_position=request.position,
            connection_id=connection_id
        )

        if result["success"]:
            return PlaybackResponse(
                success=True,
                status=result["status"],
                position=result["position"],
                message="Playback started"
            )
        else:
            return PlaybackResponse(
                success=False,
                errors=result.get("errors", ["Failed to start playback"])
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/playback/stop", response_model=PlaybackResponse)
async def stop_playback(
    project_id: uuid.UUID,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for updates")
):
    """
    Stop project playback

    Stops playback and resets position to beginning. All recording sessions
    that depend on playback sync will also be notified of the stop.
    """
    try:
        playback_service = await get_playback_service()
        result = await playback_service.stop(connection_id=connection_id)

        if result["success"]:
            return PlaybackResponse(
                success=True,
                status=result["status"],
                position=result["position"],
                message="Playback stopped"
            )
        else:
            return PlaybackResponse(
                success=False,
                errors=result.get("errors", ["Failed to stop playback"])
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/playback/pause", response_model=PlaybackResponse)
async def pause_playback(
    project_id: uuid.UUID,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for updates")
):
    """Pause project playback at current position"""
    try:
        playback_service = await get_playback_service()
        result = await playback_service.pause(connection_id=connection_id)

        return PlaybackResponse(
            success=result["success"],
            status=result.get("status"),
            position=result.get("position"),
            errors=result.get("errors", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/playback/resume", response_model=PlaybackResponse)
async def resume_playback(
    project_id: uuid.UUID,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for updates")
):
    """Resume paused playback from current position"""
    try:
        playback_service = await get_playback_service()
        result = await playback_service.resume(connection_id=connection_id)

        return PlaybackResponse(
            success=result["success"],
            status=result.get("status"),
            position=result.get("position"),
            errors=result.get("errors", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/playback/seek", response_model=PlaybackResponse)
async def seek_playback(
    project_id: uuid.UUID,
    request: PlaybackRequest,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for updates")
):
    """Seek to specific timeline position"""
    try:
        playback_service = await get_playback_service()
        result = await playback_service.seek(
            position=request.position,
            connection_id=connection_id
        )

        return PlaybackResponse(
            success=result["success"],
            status=result.get("status"),
            position=result.get("position"),
            errors=result.get("errors", [])
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_id}/playback/status", response_model=PlaybackStatusResponse)
async def get_playback_status(project_id: uuid.UUID):
    """
    Get current playback position and status

    Returns real-time playback information including current position, duration,
    playback state, and loaded project information.
    """
    try:
        playback_service = await get_playback_service()
        status = await playback_service.get_playback_status()

        return PlaybackStatusResponse(**status)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TRACK MIXING ENDPOINTS
# Real-time track mixing during playback/recording
# ============================================================================

@router.post("/tracks/{track_id}/volume")
async def set_track_volume(
    track_id: uuid.UUID,
    volume: float = Query(..., ge=0.0, le=2.0, description="Volume level (0.0 to 2.0)")
):
    """Set track volume level during playback"""
    try:
        playback_service = await get_playback_service()
        result = await playback_service.set_track_volume(track_id, volume)

        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("errors", ["Failed to set volume"]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tracks/{track_id}/mute")
async def set_track_mute(
    track_id: uuid.UUID,
    muted: bool = Query(..., description="Mute state")
):
    """Set track mute state during playback"""
    try:
        playback_service = await get_playback_service()
        result = await playback_service.set_track_mute(track_id, muted)

        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("errors", ["Failed to set mute"]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tracks/{track_id}/solo")
async def set_track_solo(
    track_id: uuid.UUID,
    soloed: bool = Query(..., description="Solo state")
):
    """Set track solo state during playback"""
    try:
        playback_service = await get_playback_service()
        result = await playback_service.set_track_solo(track_id, soloed)

        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=400, detail=result.get("errors", ["Failed to set solo"]))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ADVANCED SEPARATION ENDPOINTS
# Enhanced stem separation with rhythm/lead guitar distinction and quality enhancement
# ============================================================================

@router.post("/clips/{clip_id}/separate/advanced", response_model=dict)
async def separate_clip_advanced(
    clip_id: uuid.UUID,
    request: dict,
    background_tasks: BackgroundTasks,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for progress updates")
):
    """
    Advanced audio separation with individual instrument targeting

    Performs enhanced stem separation that can distinguish between rhythm and lead guitar,
    and provides better quality isolation of individual instruments. Supports the following
    instruments: vocals, rhythm_guitar, lead_guitar, drums, bass, piano, other.

    Example request body:
    {
        "instruments": ["vocals", "rhythm_guitar", "lead_guitar"],
        "model": "htdemucs_6s",
        "enhancement_options": {
            "quality_level": "high",
            "preview_mode": false
        }
    }
    """
    try:
        from ...services.advanced_separation_service import AdvancedSeparationService
        from ...database.schemas import AdvancedSeparationRequest

        # Validate request
        separation_request = AdvancedSeparationRequest(**request)

        # Create job for tracking
        job_id = uuid.uuid4()
        estimated_duration = len(separation_request.instruments) * 60  # Rough estimate: 1 minute per instrument

        # Start processing in background
        advanced_service = AdvancedSeparationService()

        if connection_id:
            def progress_callback(progress: float, message: str):
                # This will be called by the separation service
                pass
            advanced_service.set_progress_callback(progress_callback)

        # For now, return job creation response
        # In production, this would trigger background processing
        return {
            "job_id": str(job_id),
            "estimated_duration": estimated_duration,
            "status": "pending",
            "message": f"Advanced separation queued for {len(separation_request.instruments)} instruments",
            "target_instruments": separation_request.instruments
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced separation failed: {str(e)}")


@router.get("/jobs/{job_id}/status")
async def get_processing_job_status(job_id: uuid.UUID):
    """
    Get status of a processing job

    Returns current status, progress, and results for long-running audio processing jobs
    such as advanced separation or enhancement operations.
    """
    try:
        # TODO: Implement job status tracking with database
        # For now, return mock response
        return {
            "job_id": str(job_id),
            "status": "processing",
            "progress": 65.0,
            "current_stage": "Separating rhythm and lead guitar",
            "estimated_remaining": 45,
            "message": "Processing guitar classification..."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get job status: {str(e)}")


@router.post("/clips/{clip_id}/enhance", response_model=dict)
async def enhance_clip_stems(
    clip_id: uuid.UUID,
    request: dict,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for progress updates")
):
    """
    Enhance audio quality of separated stems

    Applies AI-powered quality enhancement to individual stems including:
    - Noise reduction and denoising
    - Clarity and presence enhancement
    - AI upsampling for higher quality
    - Dynamic range optimization

    Example request body:
    {
        "stems": ["vocals", "lead_guitar"],
        "enhancements": ["denoise", "clarity", "upsampl"],
        "level": 0.8,
        "profile_id": "optional-enhancement-profile-uuid"
    }
    """
    try:
        from ...database.schemas import EnhancementRequest

        # Validate request
        enhancement_request = EnhancementRequest(**request)

        # TODO: Implement audio enhancement service
        # For now, return mock response
        return {
            "enhanced_stems": {
                stem: {
                    "file_path": f"/enhanced/{clip_id}_{stem}_enhanced.wav",
                    "quality_score": 8.5 + (enhancement_request.level * 1.5),
                    "enhancements_applied": enhancement_request.enhancements,
                    "file_size": "2.1 MB"
                } for stem in enhancement_request.stems
            },
            "processing_time": 15.2,
            "quality_improvements": {
                stem: enhancement_request.level * 2.5 for stem in enhancement_request.stems
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@router.post("/clips/{clip_id}/enhance/preview", response_model=dict)
async def preview_enhancement(
    clip_id: uuid.UUID,
    request: dict
):
    """
    Generate a preview of audio enhancement

    Creates a short preview (typically 30 seconds) of the enhanced audio to allow
    users to test enhancement settings before applying to the full track.

    Example request body:
    {
        "stem": "vocals",
        "enhancement": "clarity",
        "level": 0.5,
        "preview_duration": 30
    }
    """
    try:
        from ...database.schemas import EnhancementPreviewRequest

        # Validate request
        preview_request = EnhancementPreviewRequest(**request)

        # TODO: Implement enhancement preview generation
        # For now, return mock response
        return {
            "preview_url": f"/api/audio/preview/{clip_id}_{preview_request.stem}_{preview_request.enhancement}.wav",
            "duration": preview_request.preview_duration,
            "quality_preview": {
                "clarity_improvement": preview_request.level * 1.8,
                "noise_reduction": preview_request.level * 2.1,
                "overall_enhancement": preview_request.level * 2.0
            },
            "expires_at": "2024-01-12T16:30:00Z"  # Preview expires in 1 hour
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")


@router.get("/clips/{clip_id}/quality-analysis", response_model=dict)
async def analyze_clip_quality(
    clip_id: uuid.UUID,
    include_stems: bool = Query(True, description="Include per-stem analysis"),
    detailed_metrics: bool = Query(False, description="Include detailed quality metrics")
):
    """
    Analyze audio quality of clip and its stems

    Provides comprehensive quality analysis including:
    - Overall quality score (0-10)
    - Per-stem quality metrics
    - Dynamic range analysis
    - Signal-to-noise ratio
    - Spectral characteristics
    """
    try:
        # TODO: Implement real quality analysis
        # For now, return mock response
        base_response = {
            "overall_score": 7.8,
            "analysis_metadata": {
                "analyzed_at": "2024-01-12T15:30:00Z",
                "analysis_duration": 2.1,
                "algorithm_version": "1.0"
            }
        }

        if include_stems:
            base_response["stems"] = {
                "vocals": {
                    "clarity": 8.2,
                    "noise_level": 0.15,
                    "dynamic_range": 9.1,
                    "frequency_response": 8.5
                },
                "guitar": {
                    "clarity": 7.8,
                    "noise_level": 0.22,
                    "dynamic_range": 8.3,
                    "frequency_response": 7.9
                },
                "drums": {
                    "clarity": 8.7,
                    "noise_level": 0.08,
                    "dynamic_range": 9.8,
                    "frequency_response": 9.2
                }
            }

        if detailed_metrics:
            base_response["detailed_metrics"] = {
                "peak_amplitude": -2.1,
                "rms_level": -18.4,
                "lufs_integrated": -16.2,
                "peak_to_loudness_ratio": 14.1,
                "stereo_width": 0.75,
                "phase_correlation": 0.92
            }

        return base_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality analysis failed: {str(e)}")


@router.get("/models/enhancement", response_model=dict)
async def get_enhancement_models():
    """
    Get available enhancement models and capabilities

    Returns information about available AI models for audio quality enhancement
    including their capabilities, supported enhancement types, and performance characteristics.
    """
    try:
        return {
            "available": [
                {
                    "name": "real_esrgan",
                    "type": "upsampling",
                    "description": "Neural upsampling for higher sample rates and bit depths",
                    "supported_formats": ["wav", "flac"],
                    "max_upsampling": "4x",
                    "processing_time_factor": 0.3
                },
                {
                    "name": "spectral_denoiser",
                    "type": "denoising",
                    "description": "AI-powered spectral noise reduction",
                    "noise_reduction_range": "up to 40dB",
                    "preserves_transients": True,
                    "processing_time_factor": 0.2
                },
                {
                    "name": "mastering_chain",
                    "type": "mastering",
                    "description": "AI-based mastering with EQ, compression, and limiting",
                    "target_lufs": [-23, -16, -14, -8],
                    "auto_eq": True,
                    "processing_time_factor": 0.1
                }
            ],
            "default_model": "mastering_chain",
            "performance_notes": {
                "cpu_usage": "Medium to High depending on model",
                "memory_requirement": "2-4GB for typical audio files",
                "gpu_acceleration": "Supported for real_esrgan model"
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get enhancement models: {str(e)}")


# ============================================================================
# ENHANCEMENT PROFILE ENDPOINTS
# User enhancement presets and preferences
# ============================================================================

@router.get("/enhancement-profiles", response_model=List[dict])
async def get_enhancement_profiles(
    user_id: uuid.UUID = Query(..., description="User ID to get profiles for"),
    include_public: bool = Query(True, description="Include public profiles")
):
    """Get user's enhancement profiles and optionally public profiles"""
    try:
        # TODO: Implement database query for enhancement profiles
        # For now, return mock response
        profiles = [
            {
                "id": str(uuid.uuid4()),
                "profile_name": "Vocal Enhancement",
                "description": "Optimized for vocal clarity and presence",
                "is_default": True,
                "is_public": False,
                "enhancement_settings": {
                    "vocals": {
                        "denoise": 0.7,
                        "clarity": 0.8,
                        "presence": 0.6
                    }
                },
                "usage_count": 45,
                "created_at": "2024-01-10T10:00:00Z"
            },
            {
                "id": str(uuid.uuid4()),
                "profile_name": "Guitar Enhancement",
                "description": "Enhances guitar separation and tone",
                "is_default": False,
                "is_public": True,
                "enhancement_settings": {
                    "rhythm_guitar": {
                        "eq": {"low": 0.1, "mid": 0.3, "high": 0.2},
                        "compression": 0.4
                    },
                    "lead_guitar": {
                        "clarity": 0.9,
                        "presence": 0.8,
                        "eq": {"low": 0.0, "mid": 0.5, "high": 0.6}
                    }
                },
                "usage_count": 23,
                "created_at": "2024-01-09T14:30:00Z"
            }
        ]

        return profiles

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get enhancement profiles: {str(e)}")


@router.post("/enhancement-profiles", response_model=dict)
async def create_enhancement_profile(request: dict):
    """Create a new enhancement profile"""
    try:
        from ...database.schemas import EnhancementProfileCreate

        # Validate request
        profile_request = EnhancementProfileCreate(**request)

        # TODO: Implement database creation
        # For now, return mock response
        profile_id = uuid.uuid4()

        return {
            "id": str(profile_id),
            "profile_name": profile_request.profile_name,
            "description": profile_request.description,
            "created_at": "2024-01-12T15:30:00Z",
            "message": "Enhancement profile created successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create enhancement profile: {str(e)}")


# ============================================================================
# BEAT GENERATION ENDPOINTS
# ============================================================================

# Import beat generation schemas
from ...database.schemas import (
    BeatGenerationRequest as BeatGenerationRequestSchema,
    BeatGenerationUpdate,
    BeatGenerationResponse,
    BeatGenerationJobResponse,
    BeatToProjectRequest,
    BeatToProjectResponse,
    BeatTemplateResponse,
    BeatVariationResponse,
    BeatGenerationProgressEvent
)


@router.post("/beats/generate", response_model=BeatGenerationJobResponse)
async def generate_beat(
    request: BeatGenerationRequestSchema,
    background_tasks: BackgroundTasks,
    connection_id: Optional[str] = Query(None, description="WebSocket connection ID for progress updates")
):
    """
    Generate AI beat with real-time progress updates

    Supports both MusicGen (local) and SOUNDRAW (API) providers.
    Processing runs in background with WebSocket progress updates.
    Beats are automatically synchronized to project tempo and time signature.
    """
    try:
        # Import services here to avoid circular imports
        from ...services.beat_generation_service import BeatGenerationService
        from ...database.repositories.beat_generation_repository import BeatGenerationRepository
        from ...core.config import get_settings

        settings = get_settings()

        # Validate provider availability
        beat_config = settings.get_beat_generation_config()
        if request.provider == "soundraw" and not beat_config["soundraw"]["enabled"]:
            raise HTTPException(
                status_code=400,
                detail="SOUNDRAW provider not configured. Set SOUNDRAW_API_KEY environment variable."
            )

        # Validate duration limits
        max_duration = beat_config["limits"]["max_duration"]
        if request.duration > max_duration:
            raise HTTPException(
                status_code=400,
                detail=f"Duration {request.duration}s exceeds maximum {max_duration}s"
            )

        # TODO: Get user_id from authentication
        user_id = uuid.uuid4()  # Placeholder

        # Create database repository (would come from dependency injection in production)
        from ...database.connection import get_session
        session = get_session()
        repo = BeatGenerationRepository(session)

        # Create beat generation request in database
        db_result = await repo.create_beat_generation_request(request, user_id)
        if not db_result.success:
            raise HTTPException(status_code=400, detail=db_result.error)

        beat_request = db_result.data

        # Start background generation task
        background_tasks.add_task(
            _process_beat_generation,
            beat_request.id,
            request.provider,
            connection_id
        )

        # Estimate duration based on provider and model
        estimated_duration = _estimate_generation_duration(request.provider, request.duration)

        return BeatGenerationJobResponse(
            request_id=beat_request.id,
            estimated_duration=estimated_duration,
            status="pending"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start beat generation: {str(e)}")


@router.get("/beats/{request_id}/status", response_model=BeatGenerationResponse)
async def get_beat_generation_status(request_id: uuid.UUID):
    """Get beat generation progress and results"""
    try:
        from ...database.repositories.beat_generation_repository import BeatGenerationRepository
        from ...database.connection import get_session

        session = get_session()
        repo = BeatGenerationRepository(session)

        result = await repo.get_beat_generation_request(request_id)
        if not result.success:
            raise HTTPException(status_code=404, detail="Beat generation request not found")

        beat_request = result.data
        if not beat_request:
            raise HTTPException(status_code=404, detail="Beat generation request not found")

        return BeatGenerationResponse(
            id=beat_request.id,
            project_id=beat_request.project_id,
            user_id=beat_request.user_id,
            prompt=beat_request.prompt,
            provider=beat_request.provider,
            model_name=beat_request.model_name,
            duration=beat_request.duration,
            tempo=beat_request.tempo,
            time_signature=beat_request.time_signature,
            style_tags=beat_request.style_tags,
            status=beat_request.status,
            progress=beat_request.progress,
            current_stage=beat_request.current_stage,
            generated_audio_path=beat_request.generated_audio_path,
            generated_midi_path=beat_request.generated_midi_path,
            quality_score=beat_request.quality_score,
            processing_time=beat_request.processing_time,
            provider_metadata=beat_request.provider_metadata,
            error_message=beat_request.error_message,
            created_at=beat_request.created_at,
            started_at=beat_request.started_at,
            completed_at=beat_request.completed_at
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get beat generation status: {str(e)}")


@router.post("/beats/{request_id}/add-to-project", response_model=BeatToProjectResponse)
async def add_beat_to_project(
    request_id: uuid.UUID,
    request: BeatToProjectRequest
):
    """Add generated beat as clip to project track"""
    try:
        from ...database.repositories.beat_generation_repository import BeatGenerationRepository
        from ...database.connection import get_session
        from ...services.project_integration_service import ProjectIntegrationService

        session = get_session()
        repo = BeatGenerationRepository(session)

        # Get beat generation request
        beat_result = await repo.get_beat_generation_request(request_id)
        if not beat_result.success or not beat_result.data:
            raise HTTPException(status_code=404, detail="Beat generation request not found")

        beat_request = beat_result.data

        # Check if generation is complete
        if beat_request.status != "completed" or not beat_request.generated_audio_path:
            raise HTTPException(
                status_code=400,
                detail="Beat generation not completed or no audio file available"
            )

        # Add beat to project
        integration_service = ProjectIntegrationService()
        result = await integration_service.add_beat_to_track(
            beat_request.generated_audio_path,
            request.track_id,
            request.timeline_position,
            request.clip_name or f"Beat - {beat_request.prompt[:30]}"
        )

        if result.success:
            return BeatToProjectResponse(
                success=True,
                clip_id=result.data.get("clip_id"),
                message="Beat added to project successfully"
            )
        else:
            return BeatToProjectResponse(
                success=False,
                message=result.error
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add beat to project: {str(e)}")


@router.get("/beats/templates", response_model=List[BeatTemplateResponse])
async def get_beat_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search in template names and tags")
):
    """Get available beat templates"""
    try:
        from ...database.repositories.beat_generation_repository import BeatGenerationRepository
        from ...database.connection import get_session

        session = get_session()
        repo = BeatGenerationRepository(session)

        # Parse search tags
        search_tags = None
        if search:
            search_tags = [tag.strip() for tag in search.lower().split(",")]

        result = await repo.get_beat_templates(
            category=category,
            is_public=True,
            search_tags=search_tags
        )

        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        templates = result.data
        template_responses = []

        for template in templates:
            template_responses.append(BeatTemplateResponse(
                id=template.id,
                name=template.name,
                description=template.description,
                category=template.category,
                tags=template.tags,
                default_tempo=template.default_tempo,
                time_signature=template.time_signature,
                duration=template.duration,
                provider_config=template.provider_config,
                prompt_template=template.prompt_template,
                is_public=template.is_public,
                usage_count=template.usage_count,
                average_quality=template.average_quality,
                created_by_user_id=template.created_by_user_id,
                created_at=template.created_at,
                updated_at=template.updated_at
            ))

        return template_responses

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get beat templates: {str(e)}")


@router.get("/beats/{request_id}/variations", response_model=List[BeatVariationResponse])
async def get_beat_variations(request_id: uuid.UUID):
    """Get all variations for a beat generation request"""
    try:
        from ...database.repositories.beat_generation_repository import BeatGenerationRepository
        from ...database.connection import get_session

        session = get_session()
        repo = BeatGenerationRepository(session)

        result = await repo.get_request_variations(request_id)
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        variations = result.data
        variation_responses = []

        for variation in variations:
            variation_responses.append(BeatVariationResponse(
                id=variation.id,
                beat_generation_request_id=variation.beat_generation_request_id,
                variation_index=variation.variation_index,
                name=variation.name,
                audio_path=variation.audio_path,
                midi_path=variation.midi_path,
                quality_score=variation.quality_score,
                user_rating=variation.user_rating,
                generation_seed=variation.generation_seed,
                generation_metadata=variation.generation_metadata,
                is_selected=variation.is_selected,
                used_in_project=variation.used_in_project,
                created_at=variation.created_at
            ))

        return variation_responses

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get beat variations: {str(e)}")


@router.post("/beats/variations/{variation_id}/select", response_model=dict)
async def select_beat_variation(variation_id: uuid.UUID):
    """Mark a variation as selected"""
    try:
        from ...database.repositories.beat_generation_repository import BeatGenerationRepository
        from ...database.connection import get_session

        session = get_session()
        repo = BeatGenerationRepository(session)

        result = await repo.select_variation(variation_id)
        if not result.success:
            raise HTTPException(status_code=400, detail=result.error)

        return {
            "success": True,
            "message": "Variation selected successfully",
            "variation_id": str(variation_id)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to select variation: {str(e)}")


@router.get("/beats/providers", response_model=dict)
async def get_beat_providers():
    """Get available beat generation providers and their capabilities"""
    try:
        from ...core.config import get_settings
        from ...services.beat_generation_service import BeatGenerationService

        settings = get_settings()
        beat_config = settings.get_beat_generation_config()
        service = BeatGenerationService()

        providers = {
            "musicgen": {
                "name": "MusicGen",
                "description": "Local AI beat generation using Meta's MusicGen models",
                "local": True,
                "available": True,  # TODO: Check actual availability
                "models": service.get_supported_models("musicgen"),
                "max_duration": beat_config["musicgen"]["max_duration"] if "musicgen" in beat_config else 30.0,
                "capabilities": ["tempo_sync", "style_conditioning", "prompt_based"]
            },
            "soundraw": {
                "name": "SOUNDRAW",
                "description": "Cloud-based professional beat generation API",
                "local": False,
                "available": beat_config["soundraw"]["enabled"],
                "models": ["soundraw-v1"],  # SOUNDRAW doesn't expose model names
                "max_duration": beat_config["soundraw"]["max_duration"],
                "capabilities": ["tempo_sync", "genre_selection", "mood_control", "instrument_selection"]
            }
        }

        return {
            "providers": providers,
            "default_provider": "musicgen",
            "global_limits": beat_config["limits"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get beat providers: {str(e)}")


# Helper functions for beat generation endpoints

async def _process_beat_generation(
    request_id: uuid.UUID,
    provider: str,
    connection_id: Optional[str] = None
):
    """Background task for beat generation processing"""
    try:
        from ...services.beat_generation_service import BeatGenerationService
        from ...database.repositories.beat_generation_repository import BeatGenerationRepository
        from ...database.connection import get_session
        from ...api.websocket import manager

        session = get_session()
        repo = BeatGenerationRepository(session)

        # Get request details
        request_result = await repo.get_beat_generation_request(request_id)
        if not request_result.success:
            return

        beat_request = request_result.data

        # Initialize service
        service = BeatGenerationService()

        # Set progress callback for WebSocket updates
        if connection_id:
            def progress_callback(progress: float, stage: str):
                asyncio.create_task(manager.send_beat_generation_progress(
                    connection_id,
                    BeatGenerationProgressEvent(
                        request_id=request_id,
                        status="processing",
                        progress=Decimal(str(progress * 100)),
                        current_stage=stage
                    )
                ))

            service.set_progress_callback(progress_callback)

        # Update status to processing
        await repo.update_beat_generation_request(
            request_id,
            BeatGenerationUpdate(status="processing", current_stage="Loading model")
        )

        # Load model
        load_result = await service.load_model(provider, beat_request.model_name)
        if not load_result.success:
            await repo.update_beat_generation_request(
                request_id,
                BeatGenerationUpdate(status="failed", error_message=load_result.error)
            )
            return

        # Generate beat
        generation_result = await service.generate_beat(
            prompt=beat_request.prompt,
            duration=float(beat_request.duration),
            tempo=float(beat_request.tempo),
            time_signature=beat_request.time_signature,
            style_tags=beat_request.style_tags,
            project_id=beat_request.project_id
        )

        if generation_result.success:
            # Save generated audio
            from ...core.config import get_settings
            settings = get_settings()

            audio_filename = f"beat_{request_id}.wav"
            audio_path = f"{settings.BEATS_PATH}/{audio_filename}"

            # TODO: Save audio array to file
            # audio_array = generation_result.data
            # await AudioFileManager.save_audio(audio_array, audio_path)

            # Update request with results
            await repo.update_beat_generation_request(
                request_id,
                BeatGenerationUpdate(
                    status="completed",
                    progress=Decimal("100.0"),
                    generated_audio_path=audio_path,
                    quality_score=generation_result.metadata.get("quality_score"),
                    processing_time=generation_result.metadata.get("processing_time_ms", 0) / 1000,
                    provider_metadata=generation_result.metadata
                )
            )

        else:
            # Update with error
            await repo.update_beat_generation_request(
                request_id,
                BeatGenerationUpdate(
                    status="failed",
                    error_message=generation_result.error
                )
            )

        # Clean up service
        await service.cleanup()

    except Exception as e:
        # Log error and update request status
        from ...core.logging import audio_logger
        audio_logger.log_processing_error(
            operation="BeatGenerationBackground",
            error=str(e),
            request_id=str(request_id)
        )


def _estimate_generation_duration(provider: str, beat_duration: float) -> int:
    """Estimate generation duration in seconds"""

    if provider == "musicgen":
        # MusicGen typically takes 1-3x real-time depending on model and hardware
        base_time = beat_duration * 2  # 2x real-time average
        return int(base_time + 30)  # Add 30s for model loading

    elif provider == "soundraw":
        # SOUNDRAW API typically takes 30-60 seconds regardless of length
        return 60

    else:
        return 120  # Default estimate