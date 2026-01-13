"""
AudioLab Audio Processing API Routes
REST endpoints for audio processing operations with real Demucs integration
Includes real-time recording and playback capabilities for overdubbing
"""

from typing import List, Optional
import uuid
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