"""
AudioLab Audio Processing Service
High-level service orchestrating audio processing workflows with real-time progress tracking
"""

import asyncio
import os
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Union
from decimal import Decimal
import numpy as np
import soundfile as sf
import librosa
from datetime import datetime

from ..services.demucs_service import DemucsService
from ..database.models import Project, Clip, StemSeparation
from ..database.schemas import StemSeparationCreate
from ..database.repositories.stem_separation_repository import StemSeparationRepository
from ..database.repositories.clip_repository import ClipRepository
from ..database.connection import get_async_session
from ..api.websocket import websocket_manager
from ..core.logging import audio_logger
from ..core.config import get_settings

settings = get_settings()


class AudioProcessingService:
    """High-level audio processing service with WebSocket integration"""

    def __init__(self):
        self.demucs_service = DemucsService()
        self._model_cache: Dict[str, bool] = {}
        self.output_directory = Path(settings.AUDIO_OUTPUT_PATH) / "stems"
        self.output_directory.mkdir(parents=True, exist_ok=True)

    async def separate_audio_clip(
        self,
        clip_id: uuid.UUID,
        model_name: str = "htdemucs_ft",
        connection_id: Optional[str] = None,
        force_reprocess: bool = False
    ) -> Dict[str, Union[str, float, Dict]]:
        """
        Separate audio clip into stems with real-time progress tracking

        Args:
            clip_id: Database ID of the clip to process
            model_name: Demucs model to use for separation
            connection_id: WebSocket connection ID for progress updates
            force_reprocess: Force reprocessing even if stems already exist

        Returns:
            Dictionary with separation results and metadata
        """
        start_time = datetime.now()

        try:
            # Setup database session and repositories
            async with get_async_session() as session:
                clip_repo = ClipRepository(session)
                stem_repo = StemSeparationRepository(session)

                # Get clip information
                clip = await clip_repo.get_or_404(clip_id)

                # Check if separation already exists
                if not force_reprocess:
                    existing = await stem_repo.get_latest_by_clip(clip_id)
                    if existing and existing.model_used == model_name:
                        if connection_id:
                            await websocket_manager.send_processing_complete(
                                connection_id,
                                "stem_separation",
                                {
                                    "clip_id": str(clip_id),
                                    "stems": existing.stems,
                                    "cached": True
                                }
                            )
                        return {
                            "success": True,
                            "clip_id": str(clip_id),
                            "stems": existing.stems,
                            "model_used": existing.model_used,
                            "processing_time": float(existing.processing_time or 0),
                            "quality_score": float(existing.quality_score or 0),
                            "cached": True
                        }

                # Setup progress callback for WebSocket updates
                progress_callback = None
                if connection_id:
                    progress_callback = websocket_manager.create_progress_callback(
                        connection_id,
                        "stem_separation"
                    )

                # Load audio file
                if connection_id:
                    await websocket_manager.send_progress_update(
                        connection_id,
                        "stem_separation",
                        0.05,
                        f"Loading audio file: {clip.name or 'Untitled'}"
                    )

                audio_data, sample_rate = await self._load_audio_file(clip.file_path)

                # Ensure model is loaded
                await self._ensure_model_loaded(model_name, progress_callback)

                # Set up progress callback in Demucs service
                if progress_callback:
                    self.demucs_service.set_progress_callback(progress_callback)

                # Run separation
                if connection_id:
                    await websocket_manager.send_progress_update(
                        connection_id,
                        "stem_separation",
                        0.15,
                        f"Starting separation with {model_name}..."
                    )

                result = await self.demucs_service._process_internal(
                    audio_data,
                    model_name=model_name,
                    sample_rate=sample_rate
                )

                if not result.success:
                    if connection_id:
                        await websocket_manager.send_processing_error(
                            connection_id,
                            "stem_separation",
                            result.error
                        )
                    return {
                        "success": False,
                        "error": result.error
                    }

                # Save stems to files
                if connection_id:
                    await websocket_manager.send_progress_update(
                        connection_id,
                        "stem_separation",
                        0.9,
                        "Saving stem files..."
                    )

                stem_files = await self._save_stems_to_files(
                    result.data,
                    clip_id,
                    model_name,
                    sample_rate
                )

                # Store results in database
                processing_time = (datetime.now() - start_time).total_seconds()

                stem_separation_data = StemSeparationCreate(
                    clip_id=clip_id,
                    stems=stem_files,
                    model_used=model_name,
                    processing_time=Decimal(str(round(processing_time, 3))),
                    quality_score=Decimal(str(result.metadata.get("quality_score", 7.5)))
                )

                stem_separation = await stem_repo.create(stem_separation_data)

                # Send completion notification
                if connection_id:
                    await websocket_manager.send_processing_complete(
                        connection_id,
                        "stem_separation",
                        {
                            "clip_id": str(clip_id),
                            "stems": stem_files,
                            "separation_id": str(stem_separation.id)
                        },
                        duration_ms=processing_time * 1000
                    )

                audio_logger.log_processing_complete(
                    operation="AudioSeparationWorkflow",
                    duration_ms=processing_time * 1000,
                    clip_id=str(clip_id),
                    model=model_name,
                    stems=list(stem_files.keys())
                )

                return {
                    "success": True,
                    "clip_id": str(clip_id),
                    "separation_id": str(stem_separation.id),
                    "stems": stem_files,
                    "model_used": model_name,
                    "processing_time": processing_time,
                    "quality_score": float(stem_separation.quality_score or 0),
                    "cached": False
                }

        except Exception as e:
            error_msg = f"Audio separation failed: {str(e)}"

            if connection_id:
                await websocket_manager.send_processing_error(
                    connection_id,
                    "stem_separation",
                    error_msg
                )

            audio_logger.log_processing_error(
                operation="AudioSeparationWorkflow",
                error=error_msg,
                clip_id=str(clip_id)
            )

            return {
                "success": False,
                "error": error_msg
            }

    async def batch_separate_clips(
        self,
        clip_ids: List[uuid.UUID],
        model_name: str = "htdemucs_ft",
        connection_id: Optional[str] = None
    ) -> Dict[str, Union[List, int]]:
        """Separate multiple clips in batch with combined progress tracking"""

        results = []
        total_clips = len(clip_ids)

        if connection_id:
            await websocket_manager.send_progress_update(
                connection_id,
                "batch_separation",
                0.0,
                f"Starting batch separation of {total_clips} clips..."
            )

        for i, clip_id in enumerate(clip_ids):
            # Update batch progress
            batch_progress = i / total_clips
            if connection_id:
                await websocket_manager.send_progress_update(
                    connection_id,
                    "batch_separation",
                    batch_progress,
                    f"Processing clip {i+1}/{total_clips}..."
                )

            # Process individual clip (no individual WebSocket updates)
            result = await self.separate_audio_clip(
                clip_id,
                model_name=model_name,
                connection_id=None,  # No individual progress for batch
                force_reprocess=False
            )

            results.append({
                "clip_id": str(clip_id),
                **result
            })

        # Send batch completion
        if connection_id:
            successful = sum(1 for r in results if r.get("success", False))
            await websocket_manager.send_processing_complete(
                connection_id,
                "batch_separation",
                {
                    "total_clips": total_clips,
                    "successful": successful,
                    "failed": total_clips - successful,
                    "results": results
                }
            )

        return {
            "success": True,
            "total_clips": total_clips,
            "results": results,
            "successful_count": sum(1 for r in results if r.get("success", False))
        }

    async def _load_audio_file(self, file_path: str) -> tuple[np.ndarray, int]:
        """Load audio file and convert to numpy array"""
        try:
            # Use librosa for robust audio loading
            audio, sample_rate = librosa.load(file_path, sr=None, mono=False)

            # Ensure stereo format
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            elif audio.ndim == 2 and audio.shape[0] > 2:
                audio = audio[:2]  # Take first two channels

            return audio.astype(np.float32), sample_rate

        except Exception as e:
            raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")

    async def _ensure_model_loaded(
        self,
        model_name: str,
        progress_callback: Optional[callable] = None
    ):
        """Ensure Demucs model is loaded, load if necessary"""
        if model_name not in self._model_cache or not self._model_cache[model_name]:
            if progress_callback:
                self.demucs_service.set_progress_callback(progress_callback)

            await self.demucs_service.load_model(model_name)
            self._model_cache[model_name] = True

    async def _save_stems_to_files(
        self,
        stems: Dict[str, np.ndarray],
        clip_id: uuid.UUID,
        model_name: str,
        sample_rate: int
    ) -> Dict[str, str]:
        """Save separated stems to individual audio files"""

        # Create unique directory for this separation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem_dir = self.output_directory / f"{clip_id}_{model_name}_{timestamp}"
        stem_dir.mkdir(parents=True, exist_ok=True)

        stem_files = {}

        for stem_name, stem_audio in stems.items():
            # Generate filename
            filename = f"{stem_name}.wav"
            file_path = stem_dir / filename

            # Convert to proper format for saving (channels x samples -> samples x channels)
            if stem_audio.ndim == 2:
                stem_audio_for_save = stem_audio.T
            else:
                stem_audio_for_save = stem_audio

            # Save using soundfile for high quality
            sf.write(
                str(file_path),
                stem_audio_for_save,
                sample_rate,
                format='WAV',
                subtype='PCM_24'  # 24-bit for professional quality
            )

            # Store relative path from output directory
            relative_path = str(file_path.relative_to(self.output_directory.parent))
            stem_files[stem_name] = relative_path

        return stem_files

    async def get_available_models(self) -> List[Dict[str, Union[str, List[str]]]]:
        """Get list of available Demucs models and their capabilities"""
        models = []

        for model_name, stems in DemucsService.SUPPORTED_MODELS.items():
            models.append({
                "name": model_name,
                "stems": stems,
                "description": self._get_model_description(model_name),
                "loaded": self._model_cache.get(model_name, False)
            })

        return models

    def _get_model_description(self, model_name: str) -> str:
        """Get human-readable description of model"""
        descriptions = {
            "htdemucs": "Hybrid Demucs - Balanced quality and speed",
            "htdemucs_ft": "Hybrid Demucs Fine-tuned - Enhanced separation quality",
            "htdemucs_6s": "6-Source Demucs - Separates vocals, drums, bass, other, piano, guitar",
            "mdx_extra": "MDX Extra - Optimized for vocal extraction",
            "mdx_q": "MDX Quantized - Fast processing with good quality"
        }
        return descriptions.get(model_name, "Demucs model for audio stem separation")

    async def cleanup(self):
        """Clean up resources"""
        await self.demucs_service.cleanup()
        self._model_cache.clear()


# Global service instance
audio_processing_service = AudioProcessingService()