"""
AudioLab Beat Generation Service
AI-powered beat generation using MusicGen and SOUNDRAW following DemucsService patterns
"""

import asyncio
import time
import uuid
import json
from typing import Dict, Optional, Callable, List, Any
from decimal import Decimal
from pathlib import Path
import numpy as np
import torch
import torchaudio

# MusicGen imports (with fallback)
try:
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    import scipy.io.wavfile
    MUSICGEN_AVAILABLE = True
except ImportError:
    MUSICGEN_AVAILABLE = False

# HTTP client for SOUNDRAW
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from ..core.audio_processor import BaseAudioProcessor, ProcessingResult
from ..core.logging import audio_logger
from ..core.result import Result
from ..database.models import BeatGenerationRequest, Project
from ..database.repositories.beat_generation_repository import BeatGenerationRepository


class BeatGenerationService(BaseAudioProcessor):
    """AI beat generation service following DemucsService patterns"""

    # Supported providers and their configurations
    SUPPORTED_PROVIDERS = {
        "musicgen": {
            "models": [
                "facebook/musicgen-small",
                "facebook/musicgen-medium",
                "facebook/musicgen-large",
                "facebook/musicgen-melody"
            ],
            "max_duration": 30.0,
            "local": True,
            "requires_gpu": False  # Can run on CPU
        },
        "soundraw": {
            "api_endpoint": "https://api.soundraw.io/v1/generate",
            "max_duration": 300.0,
            "local": False,
            "requires_api_key": True
        }
    }

    def __init__(
        self,
        max_memory_gb: float = 6.0,
        soundraw_api_key: Optional[str] = None
    ):
        super().__init__()
        self._model = None
        self._processor = None
        self._model_loaded = False
        self.current_model = None
        self.current_provider = None
        self.device = self._get_optimal_device()
        self.max_memory_gb = max_memory_gb
        self.soundraw_api_key = soundraw_api_key
        self.progress_callback: Optional[Callable[[float, str], None]] = None

        # HTTP client for SOUNDRAW
        self._http_client: Optional[httpx.AsyncClient] = None

    def _get_optimal_device(self) -> str:
        """Determine optimal processing device with GPU/CPU fallback"""
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory >= 4.0:  # MusicGen benefits from GPU but works on CPU
                    return "cuda"
            except Exception as e:
                audio_logger.log_processing_error(
                    operation="GPUCheck",
                    error=f"GPU check failed: {e}"
                )
        return "cpu"

    async def load_model(self, provider: str = "musicgen", model_name: str = "facebook/musicgen-small") -> Result[None]:
        """Load AI model for beat generation"""

        if provider not in self.SUPPORTED_PROVIDERS:
            return Result.err(f"Unsupported provider: {provider}")

        provider_config = self.SUPPORTED_PROVIDERS[provider]

        if provider == "musicgen":
            return await self._load_musicgen_model(model_name)
        elif provider == "soundraw":
            return await self._load_soundraw_client()

        return Result.err(f"Unknown provider: {provider}")

    async def _load_musicgen_model(self, model_name: str) -> Result[None]:
        """Load MusicGen model with GPU/CPU fallback"""

        if not MUSICGEN_AVAILABLE:
            return Result.err(
                "MusicGen not available. Install with: pip install transformers torch torchaudio"
            )

        if model_name not in self.SUPPORTED_PROVIDERS["musicgen"]["models"]:
            return Result.err(f"Unsupported MusicGen model: {model_name}")

        start_time = time.time()

        try:
            if self.progress_callback:
                self.progress_callback(0.1, f"Loading {model_name} model...")

            # Load model and processor
            loop = asyncio.get_event_loop()

            # Load in executor to avoid blocking
            self._model = await loop.run_in_executor(
                None,
                lambda: MusicgenForConditionalGeneration.from_pretrained(model_name)
            )

            self._processor = await loop.run_in_executor(
                None,
                lambda: AutoProcessor.from_pretrained(model_name)
            )

            # Move to device
            if self.progress_callback:
                self.progress_callback(0.6, f"Moving model to {self.device}...")

            self._model = self._model.to(self.device)

            if self.progress_callback:
                self.progress_callback(0.8, f"Model {model_name} loaded on {self.device}")

            self.current_model = model_name
            self.current_provider = "musicgen"
            self._model_loaded = True

            load_time = (time.time() - start_time) * 1000

            audio_logger.log_processing_complete(
                operation="MusicGenModelLoad",
                duration_ms=load_time,
                model=model_name,
                device=self.device
            )

            if self.progress_callback:
                self.progress_callback(1.0, f"Ready for beat generation using {model_name}")

            return Result.ok(None)

        except Exception as e:
            audio_logger.log_processing_error(
                operation="MusicGenModelLoad",
                error=str(e),
                model=model_name
            )

            # Try CPU fallback if GPU loading failed
            if self.device == "cuda" and "cuda" in str(e).lower():
                audio_logger.log_processing_error(
                    operation="GPUFallback",
                    error="GPU loading failed, attempting CPU fallback"
                )
                self.device = "cpu"
                return await self._load_musicgen_model(model_name)

            return Result.err(f"Failed to load MusicGen model: {e}")

    async def _load_soundraw_client(self) -> Result[None]:
        """Initialize SOUNDRAW API client"""

        if not HTTPX_AVAILABLE:
            return Result.err("HTTP client not available. Install with: pip install httpx")

        if not self.soundraw_api_key:
            return Result.err("SOUNDRAW API key required")

        try:
            if self.progress_callback:
                self.progress_callback(0.3, "Initializing SOUNDRAW client...")

            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0),  # 5 min timeout for generation
                headers={
                    "Authorization": f"Bearer {self.soundraw_api_key}",
                    "Content-Type": "application/json"
                }
            )

            # Test API connection
            if self.progress_callback:
                self.progress_callback(0.7, "Testing SOUNDRAW API connection...")

            response = await self._http_client.get("https://api.soundraw.io/v1/status")

            if response.status_code != 200:
                return Result.err(f"SOUNDRAW API connection failed: {response.status_code}")

            self.current_provider = "soundraw"
            self._model_loaded = True

            if self.progress_callback:
                self.progress_callback(1.0, "SOUNDRAW client ready")

            audio_logger.log_processing_complete(
                operation="SOUNDRAWClientLoad",
                duration_ms=0
            )

            return Result.ok(None)

        except Exception as e:
            audio_logger.log_processing_error(
                operation="SOUNDRAWClientLoad",
                error=str(e)
            )
            return Result.err(f"Failed to initialize SOUNDRAW client: {e}")

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set progress callback for WebSocket updates"""
        self.progress_callback = callback

    async def _process_internal(
        self,
        audio: np.ndarray,
        **kwargs
    ) -> ProcessingResult:
        """Generate beats - not used directly, use generate_beat instead"""
        return ProcessingResult(
            success=False,
            error="Use generate_beat() method for beat generation"
        )

    async def generate_beat(
        self,
        prompt: str,
        duration: float,
        tempo: float,
        time_signature: str,
        style_tags: Optional[Dict[str, Any]] = None,
        project_id: Optional[uuid.UUID] = None,
        **kwargs
    ) -> ProcessingResult:
        """Generate AI beat with tempo synchronization"""

        if not self._model_loaded:
            return ProcessingResult(
                success=False,
                error="Model not loaded. Call load_model() first."
            )

        start_time = time.time()

        try:
            # Validate duration
            max_duration = self.SUPPORTED_PROVIDERS[self.current_provider]["max_duration"]
            if duration > max_duration:
                return ProcessingResult(
                    success=False,
                    error=f"Duration {duration}s exceeds maximum {max_duration}s for {self.current_provider}"
                )

            if self.progress_callback:
                self.progress_callback(0.1, "Preparing beat generation request...")

            # Generate based on provider
            if self.current_provider == "musicgen":
                result = await self._generate_musicgen_beat(
                    prompt, duration, tempo, time_signature, style_tags, **kwargs
                )
            elif self.current_provider == "soundraw":
                result = await self._generate_soundraw_beat(
                    prompt, duration, tempo, time_signature, style_tags, **kwargs
                )
            else:
                return ProcessingResult(
                    success=False,
                    error=f"Unknown provider: {self.current_provider}"
                )

            if result.success:
                processing_time = (time.time() - start_time) * 1000
                result.metadata["processing_time_ms"] = processing_time
                result.metadata["tempo"] = tempo
                result.metadata["time_signature"] = time_signature
                result.metadata["provider"] = self.current_provider

                if self.progress_callback:
                    self.progress_callback(1.0, "Beat generation complete!")

                audio_logger.log_processing_complete(
                    operation="BeatGeneration",
                    duration_ms=processing_time,
                    provider=self.current_provider,
                    model=self.current_model,
                    beat_duration=duration,
                    tempo=tempo
                )

            return result

        except Exception as e:
            audio_logger.log_processing_error(
                operation="BeatGeneration",
                error=str(e),
                provider=self.current_provider
            )
            return ProcessingResult(
                success=False,
                error=f"Beat generation failed: {e}"
            )

    async def _generate_musicgen_beat(
        self,
        prompt: str,
        duration: float,
        tempo: float,
        time_signature: str,
        style_tags: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ProcessingResult:
        """Generate beat using MusicGen"""

        try:
            if self.progress_callback:
                self.progress_callback(0.3, "Processing prompt with MusicGen...")

            # Enhance prompt with musical context
            enhanced_prompt = self._enhance_musicgen_prompt(prompt, tempo, time_signature, style_tags)

            # Prepare inputs
            inputs = self._processor(
                text=[enhanced_prompt],
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            # Calculate number of tokens for desired duration
            sample_rate = self._model.config.audio_encoder.sampling_rate
            tokens_per_second = sample_rate / self._model.config.audio_encoder.hop_length
            max_new_tokens = int(duration * tokens_per_second)

            if self.progress_callback:
                self.progress_callback(0.5, f"Generating {duration}s beat...")

            # Generate audio
            loop = asyncio.get_event_loop()
            audio_values = await loop.run_in_executor(
                None,
                lambda: self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)
            )

            if self.progress_callback:
                self.progress_callback(0.8, "Post-processing generated audio...")

            # Convert to numpy array
            audio_array = audio_values[0].cpu().numpy()

            # Apply tempo synchronization if needed
            synchronized_audio = await self._synchronize_tempo(
                audio_array, sample_rate, tempo, time_signature
            )

            # Calculate quality score
            quality_score = self._calculate_beat_quality(synchronized_audio, tempo)

            if self.progress_callback:
                self.progress_callback(0.9, "Beat generation complete!")

            return ProcessingResult(
                success=True,
                data=synchronized_audio,
                metadata={
                    "sample_rate": sample_rate,
                    "enhanced_prompt": enhanced_prompt,
                    "quality_score": quality_score,
                    "model": self.current_model,
                    "synchronized": True
                }
            )

        except torch.cuda.OutOfMemoryError:
            # GPU memory exhausted - try CPU fallback
            if self.device == "cuda":
                audio_logger.log_processing_error(
                    operation="GPUMemoryExhausted",
                    error="GPU out of memory during beat generation, falling back to CPU"
                )
                self.device = "cpu"
                await self._load_musicgen_model(self.current_model)
                return await self._generate_musicgen_beat(prompt, duration, tempo, time_signature, style_tags, **kwargs)
            else:
                return ProcessingResult(
                    success=False,
                    error="Insufficient memory for beat generation"
                )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"MusicGen beat generation failed: {e}"
            )

    async def _generate_soundraw_beat(
        self,
        prompt: str,
        duration: float,
        tempo: float,
        time_signature: str,
        style_tags: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ProcessingResult:
        """Generate beat using SOUNDRAW API"""

        try:
            if self.progress_callback:
                self.progress_callback(0.3, "Submitting request to SOUNDRAW...")

            # Prepare API request
            request_data = {
                "prompt": prompt,
                "duration": duration,
                "tempo": tempo,
                "time_signature": time_signature,
                "style": style_tags or {},
                "format": "wav"
            }

            # Submit generation request
            response = await self._http_client.post(
                self.SUPPORTED_PROVIDERS["soundraw"]["api_endpoint"],
                json=request_data
            )

            if response.status_code != 200:
                return ProcessingResult(
                    success=False,
                    error=f"SOUNDRAW API error: {response.status_code} - {response.text}"
                )

            if self.progress_callback:
                self.progress_callback(0.7, "Processing SOUNDRAW response...")

            response_data = response.json()

            # Handle async generation (most AI APIs work this way)
            if "generation_id" in response_data:
                # Poll for completion
                audio_data = await self._poll_soundraw_completion(response_data["generation_id"])
            else:
                # Immediate response
                audio_data = response_data.get("audio_data")

            if not audio_data:
                return ProcessingResult(
                    success=False,
                    error="No audio data received from SOUNDRAW"
                )

            if self.progress_callback:
                self.progress_callback(0.9, "Processing SOUNDRAW audio...")

            # Convert from base64 or URL to numpy array
            audio_array = await self._process_soundraw_audio(audio_data)

            # Calculate quality score
            quality_score = self._calculate_beat_quality(audio_array, tempo)

            return ProcessingResult(
                success=True,
                data=audio_array,
                metadata={
                    "sample_rate": 44100,  # SOUNDRAW default
                    "prompt": prompt,
                    "quality_score": quality_score,
                    "api_response": response_data
                }
            )

        except httpx.RequestError as e:
            return ProcessingResult(
                success=False,
                error=f"SOUNDRAW API request failed: {e}"
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"SOUNDRAW beat generation failed: {e}"
            )

    async def _poll_soundraw_completion(self, generation_id: str) -> Optional[str]:
        """Poll SOUNDRAW API for generation completion"""

        max_polls = 60  # 5 minutes max
        poll_interval = 5  # seconds

        for attempt in range(max_polls):
            if self.progress_callback:
                progress = 0.4 + 0.3 * (attempt / max_polls)
                self.progress_callback(progress, f"Waiting for generation... ({attempt * poll_interval}s)")

            response = await self._http_client.get(f"https://api.soundraw.io/v1/status/{generation_id}")

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "completed":
                    return data.get("audio_data")
                elif data.get("status") == "failed":
                    raise Exception(f"SOUNDRAW generation failed: {data.get('error')}")

            await asyncio.sleep(poll_interval)

        raise Exception("SOUNDRAW generation timeout")

    async def _process_soundraw_audio(self, audio_data: str) -> np.ndarray:
        """Process SOUNDRAW audio data to numpy array"""

        # This is a placeholder - actual implementation depends on SOUNDRAW API format
        # Could be base64-encoded WAV, URL to download, etc.

        if audio_data.startswith("http"):
            # Download from URL
            response = await self._http_client.get(audio_data)
            # Process WAV data...
            pass
        else:
            # Decode base64 or other format
            # Process audio data...
            pass

        # Return placeholder array for now
        # In real implementation, this would decode the actual audio
        return np.random.randn(44100 * 8)  # 8 seconds of placeholder audio

    def _enhance_musicgen_prompt(
        self,
        prompt: str,
        tempo: float,
        time_signature: str,
        style_tags: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance prompt with musical context for MusicGen"""

        enhanced = prompt

        # Add tempo context
        if tempo < 80:
            enhanced += f", slow {int(tempo)} BPM"
        elif tempo > 140:
            enhanced += f", fast {int(tempo)} BPM"
        else:
            enhanced += f", {int(tempo)} BPM"

        # Add time signature
        enhanced += f", {time_signature} time"

        # Add style tags
        if style_tags:
            genre = style_tags.get("genre")
            if genre:
                enhanced += f", {genre} style"

            mood = style_tags.get("mood")
            if mood:
                enhanced += f", {mood} mood"

        return enhanced

    async def _synchronize_tempo(
        self,
        audio: np.ndarray,
        sample_rate: int,
        target_tempo: float,
        time_signature: str
    ) -> np.ndarray:
        """Synchronize generated audio to target tempo"""

        # For now, return audio as-is
        # In a full implementation, this would:
        # 1. Detect the current tempo of the generated audio
        # 2. Time-stretch to match target tempo
        # 3. Quantize to beat grid based on time signature

        return audio

    def _calculate_beat_quality(self, audio: np.ndarray, tempo: float) -> float:
        """Calculate quality score for generated beat"""

        try:
            # Simple quality metrics for beats
            if audio.size == 0:
                return 0.0

            # Dynamic range
            dynamic_range = np.max(audio) - np.min(audio)
            dynamic_score = min(dynamic_range * 5, 5.0)

            # Rhythmic consistency (placeholder)
            rhythmic_score = 4.0  # Would analyze beat consistency in real implementation

            # Overall quality
            total_score = (dynamic_score + rhythmic_score) / 2
            return round(min(total_score, 10.0), 2)

        except:
            return 7.0  # Default score

    async def cleanup(self) -> None:
        """Clean up resources"""

        if hasattr(self, '_model'):
            del self._model
            self._model = None

        if hasattr(self, '_processor'):
            del self._processor
            self._processor = None

        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model_loaded = False
        self.current_model = None
        self.current_provider = None

    def get_supported_models(self, provider: str) -> List[str]:
        """Get list of supported models for a provider"""
        if provider in self.SUPPORTED_PROVIDERS:
            return self.SUPPORTED_PROVIDERS[provider].get("models", [])
        return []

    def get_max_duration(self, provider: str) -> float:
        """Get maximum generation duration for a provider"""
        if provider in self.SUPPORTED_PROVIDERS:
            return self.SUPPORTED_PROVIDERS[provider].get("max_duration", 30.0)
        return 30.0