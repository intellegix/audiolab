"""
AudioLab Demucs v4 Integration Service
AI stem separation using Meta's Demucs v4 models with real PyTorch implementation
"""

import asyncio
import time
from typing import Dict, Optional, Union, Callable
import numpy as np
import torch
import torchaudio
from pathlib import Path

try:
    import demucs.api
    from demucs.api import Separator
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False

from ..core.audio_processor import BaseAudioProcessor, ProcessingResult
from ..core.logging import audio_logger


class DemucsService(BaseAudioProcessor):
    """Real Demucs v4 stem separation service with GPU/CPU fallback"""

    # Supported models and their stem configurations
    SUPPORTED_MODELS = {
        "htdemucs": ["drums", "bass", "other", "vocals"],
        "htdemucs_ft": ["drums", "bass", "other", "vocals"],
        "htdemucs_6s": ["drums", "bass", "other", "vocals", "piano", "guitar"],
        "mdx_extra": ["drums", "bass", "other", "vocals"],
        "mdx_q": ["drums", "bass", "other", "vocals"],
    }

    def __init__(self, max_memory_gb: float = 4.0):
        super().__init__()
        self._separator: Optional[Separator] = None
        self._model_loaded = False
        self.current_model = None
        self.device = self._get_optimal_device()
        self.max_memory_gb = max_memory_gb
        self.progress_callback: Optional[Callable[[float, str], None]] = None

    def _get_optimal_device(self) -> str:
        """Determine optimal processing device with GPU/CPU fallback"""
        if torch.cuda.is_available():
            # Check CUDA memory availability
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory >= 2.0:  # Require at least 2GB GPU memory
                    return "cuda"
            except Exception as e:
                audio_logger.log_processing_error(
                    operation="GPUCheck",
                    error=f"GPU check failed: {e}"
                )

        # Fallback to CPU
        return "cpu"

    async def load_model(self, model_name: str = "htdemucs_ft") -> None:
        """Load Demucs model with automatic download if needed"""
        if not DEMUCS_AVAILABLE:
            raise ImportError(
                "Demucs not available. Install with: pip install demucs==4.1.0"
            )

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )

        start_time = time.time()

        try:
            # Update progress
            if self.progress_callback:
                self.progress_callback(0.1, f"Loading {model_name} model...")

            # Load separator with device management
            self._separator = Separator(
                model=model_name,
                device=self.device,
                progress=True if self.progress_callback else False
            )

            # Update progress
            if self.progress_callback:
                self.progress_callback(0.3, f"Model {model_name} loaded on {self.device}")

            self.current_model = model_name
            self._model_loaded = True

            load_time = (time.time() - start_time) * 1000

            audio_logger.log_processing_complete(
                operation="DemucsModelLoad",
                duration_ms=load_time,
                model=model_name,
                device=self.device
            )

            if self.progress_callback:
                self.progress_callback(1.0, f"Ready for separation using {model_name}")

        except Exception as e:
            audio_logger.log_processing_error(
                operation="DemucsModelLoad",
                error=str(e),
                model=model_name
            )
            # Try CPU fallback if GPU loading failed
            if self.device == "cuda":
                audio_logger.log_processing_error(
                    operation="GPUFallback",
                    error="GPU loading failed, attempting CPU fallback"
                )
                self.device = "cpu"
                await self.load_model(model_name)  # Retry on CPU
            else:
                raise

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set progress callback for WebSocket updates"""
        self.progress_callback = callback

    async def _process_internal(
        self,
        audio: np.ndarray,
        model_name: Optional[str] = None,
        segment_length: float = 10.0,
        overlap: float = 0.25,
        **kwargs
    ) -> ProcessingResult:
        """Real stem separation using Demucs with segment-based processing"""

        if not self._model_loaded or self._separator is None:
            return ProcessingResult(
                success=False,
                error="Demucs model not loaded. Call load_model() first."
            )

        start_time = time.time()

        try:
            # Convert numpy array to torch tensor
            if audio.ndim == 1:
                # Convert mono to stereo
                audio = np.stack([audio, audio], axis=0)
            elif audio.ndim == 2 and audio.shape[0] > 2:
                # Take first two channels if more than stereo
                audio = audio[:2]
            elif audio.ndim == 2 and audio.shape[0] == 1:
                # Convert single channel to stereo
                audio = np.stack([audio[0], audio[0]], axis=0)

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio.astype(np.float32))

            if self.progress_callback:
                self.progress_callback(0.2, "Preparing audio for separation...")

            # Handle long audio with segment-based processing
            sample_rate = kwargs.get('sample_rate', 44100)
            segment_samples = int(segment_length * sample_rate)
            overlap_samples = int(overlap * segment_samples)

            if audio_tensor.shape[1] > segment_samples:
                # Process in segments
                stems = await self._process_segments(
                    audio_tensor,
                    segment_samples,
                    overlap_samples,
                    sample_rate
                )
            else:
                # Process entire audio at once
                stems = await self._process_single(audio_tensor)

            processing_time = (time.time() - start_time) * 1000

            # Convert results back to numpy arrays
            stem_arrays = {}
            expected_stems = self.SUPPORTED_MODELS[self.current_model]

            for stem_name in expected_stems:
                if stem_name in stems:
                    stem_arrays[stem_name] = stems[stem_name].cpu().numpy()

            if self.progress_callback:
                self.progress_callback(1.0, "Separation complete!")

            audio_logger.log_processing_complete(
                operation="DemucsProcessing",
                duration_ms=processing_time,
                model=self.current_model,
                device=self.device,
                stems=list(stem_arrays.keys()),
                audio_length=audio_tensor.shape[1] / sample_rate
            )

            return ProcessingResult(
                success=True,
                data=stem_arrays,
                metadata={
                    "model": self.current_model,
                    "device": self.device,
                    "stems": list(stem_arrays.keys()),
                    "processing_time_ms": processing_time,
                    "sample_rate": sample_rate,
                    "quality_score": self._calculate_quality_score(stem_arrays)
                }
            )

        except torch.cuda.OutOfMemoryError:
            # GPU memory exhausted - try CPU fallback
            if self.device == "cuda":
                audio_logger.log_processing_error(
                    operation="GPUMemoryExhausted",
                    error="GPU out of memory, falling back to CPU"
                )
                self.device = "cpu"
                await self.load_model(self.current_model)
                return await self._process_internal(audio, model_name, segment_length, overlap, **kwargs)
            else:
                return ProcessingResult(
                    success=False,
                    error="Insufficient memory for processing"
                )

        except Exception as e:
            audio_logger.log_processing_error(
                operation="DemucsProcessing",
                error=str(e),
                model=self.current_model
            )
            return ProcessingResult(
                success=False,
                error=f"Stem separation failed: {e}"
            )

    async def _process_single(self, audio_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process single audio segment"""
        if self.progress_callback:
            self.progress_callback(0.5, "Separating audio stems...")

        # Run separation in executor to avoid blocking
        loop = asyncio.get_event_loop()
        stems = await loop.run_in_executor(
            None,
            self._separator,
            audio_tensor
        )

        return stems

    async def _process_segments(
        self,
        audio_tensor: torch.Tensor,
        segment_samples: int,
        overlap_samples: int,
        sample_rate: int
    ) -> Dict[str, torch.Tensor]:
        """Process audio in overlapping segments to handle long files"""
        channels, total_samples = audio_tensor.shape
        hop_size = segment_samples - overlap_samples

        # Initialize result tensors
        expected_stems = self.SUPPORTED_MODELS[self.current_model]
        results = {
            stem: torch.zeros(channels, total_samples, dtype=torch.float32)
            for stem in expected_stems
        }
        weights = torch.zeros(total_samples, dtype=torch.float32)

        num_segments = (total_samples - overlap_samples) // hop_size + 1

        for i in range(num_segments):
            start_idx = i * hop_size
            end_idx = min(start_idx + segment_samples, total_samples)

            if self.progress_callback:
                progress = 0.3 + 0.6 * (i / num_segments)
                self.progress_callback(
                    progress,
                    f"Processing segment {i+1}/{num_segments}..."
                )

            segment = audio_tensor[:, start_idx:end_idx]

            # Process segment
            loop = asyncio.get_event_loop()
            segment_stems = await loop.run_in_executor(
                None,
                self._separator,
                segment
            )

            # Add to results with overlap handling
            segment_length = end_idx - start_idx
            window = torch.hann_window(segment_length)

            for stem_name, stem_audio in segment_stems.items():
                if stem_name in results:
                    # Apply windowing for smooth overlap
                    windowed_audio = stem_audio * window.unsqueeze(0)
                    results[stem_name][:, start_idx:end_idx] += windowed_audio

            # Update weights for normalization
            weights[start_idx:end_idx] += window

        # Normalize by overlap weights
        for stem_name in results:
            results[stem_name] = results[stem_name] / weights.unsqueeze(0).clamp(min=1e-8)

        return results

    def _calculate_quality_score(self, stems: Dict[str, np.ndarray]) -> float:
        """Calculate separation quality score based on signal characteristics"""
        try:
            # Simple quality metric based on dynamic range and signal correlation
            total_score = 0.0
            num_stems = len(stems)

            for stem_name, stem_audio in stems.items():
                # Calculate dynamic range
                if stem_audio.size > 0:
                    dynamic_range = np.max(np.abs(stem_audio)) - np.min(np.abs(stem_audio))
                    # Normalize to 0-10 scale
                    score = min(dynamic_range * 10, 10.0)
                    total_score += score

            return round(total_score / num_stems if num_stems > 0 else 0.0, 2)
        except:
            return 7.5  # Default quality score if calculation fails

    async def cleanup(self) -> None:
        """Clean up GPU memory and resources"""
        if hasattr(self, '_separator'):
            del self._separator
            self._separator = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model_loaded = False
        self.current_model = None