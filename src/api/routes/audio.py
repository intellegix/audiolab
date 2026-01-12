"""
AudioLab Audio Processing API Routes
REST endpoints for audio processing operations with real Demucs integration
"""

from typing import List, Optional
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

from ...core.audio_processor import AudioFileManager, ProcessingResult
from ...services.audio_processing_service import audio_processing_service

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