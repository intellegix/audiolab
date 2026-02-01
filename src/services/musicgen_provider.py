"""
MusicGen Provider for AudioLab Beat Generation
Local PyTorch-based AI beat generation using Meta's MusicGen models
"""

import asyncio
import time
import numpy as np
import torch
import torchaudio
from typing import Dict, Optional, Any, List
from pathlib import Path

try:
    from transformers import MusicgenForConditionalGeneration, AutoProcessor
    MUSICGEN_AVAILABLE = True
except ImportError:
    MUSICGEN_AVAILABLE = False

from ..core.audio_processor import ProcessingResult
from ..core.logging import audio_logger
from ..core.result import Result


class MusicGenProvider:
    """MusicGen provider for local AI beat generation"""

    SUPPORTED_MODELS = {
        "facebook/musicgen-small": {
            "description": "Small model, fast generation, 300M parameters",
            "memory_gb": 1.5,
            "quality": "good",
            "speed": "fast"
        },
        "facebook/musicgen-medium": {
            "description": "Medium model, balanced quality/speed, 1.5B parameters",
            "memory_gb": 3.0,
            "quality": "better",
            "speed": "medium"
        },
        "facebook/musicgen-large": {
            "description": "Large model, high quality, 3.3B parameters",
            "memory_gb": 6.0,
            "quality": "best",
            "speed": "slow"
        },
        "facebook/musicgen-melody": {
            "description": "Melody-conditioned model, supports musical conditioning",
            "memory_gb": 3.0,
            "quality": "better",
            "speed": "medium"
        }
    }

    MAX_DURATION = 30.0  # seconds
    SAMPLE_RATE = 32000  # MusicGen's native sample rate

    def __init__(self, device: Optional[str] = None):
        self.device = device or self._get_optimal_device()
        self._model = None
        self._processor = None
        self.current_model = None
        self._model_loaded = False

    def _get_optimal_device(self) -> str:
        """Determine optimal device for MusicGen"""
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if gpu_memory >= 2.0:  # Minimum for small model
                    return "cuda"
            except Exception:
                pass
        return "cpu"

    async def load_model(self, model_name: str) -> Result[None]:
        """Load MusicGen model"""

        if not MUSICGEN_AVAILABLE:
            return Result.err(
                "MusicGen dependencies not available. Install with: "
                "pip install transformers torch torchaudio"
            )

        if model_name not in self.SUPPORTED_MODELS:
            return Result.err(
                f"Unsupported model: {model_name}. "
                f"Supported: {list(self.SUPPORTED_MODELS.keys())}"
            )

        # Check memory requirements
        required_memory = self.SUPPORTED_MODELS[model_name]["memory_gb"]
        if self.device == "cuda":
            try:
                available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if available_memory < required_memory:
                    audio_logger.log_processing_error(
                        operation="MusicGenMemoryCheck",
                        error=f"Insufficient GPU memory: {available_memory:.1f}GB < {required_memory}GB"
                    )
                    # Fall back to CPU
                    self.device = "cpu"
            except Exception:
                self.device = "cpu"

        start_time = time.time()

        try:
            # Load model and processor in executor
            loop = asyncio.get_event_loop()

            audio_logger.log_processing_start(
                operation="MusicGenModelLoad",
                model=model_name,
                device=self.device
            )

            self._model = await loop.run_in_executor(
                None,
                lambda: MusicgenForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
            )

            self._processor = await loop.run_in_executor(
                None,
                lambda: AutoProcessor.from_pretrained(model_name)
            )

            # Move to device
            self._model = self._model.to(self.device)

            # Enable optimizations
            if self.device == "cuda":
                self._model = torch.compile(self._model)

            self.current_model = model_name
            self._model_loaded = True

            load_time = time.time() - start_time

            audio_logger.log_processing_complete(
                operation="MusicGenModelLoad",
                duration_ms=load_time * 1000,
                model=model_name,
                device=self.device
            )

            return Result.ok(None)

        except Exception as e:
            audio_logger.log_processing_error(
                operation="MusicGenModelLoad",
                error=str(e),
                model=model_name
            )
            return Result.err(f"Failed to load MusicGen model: {e}")

    async def generate(
        self,
        prompt: str,
        duration: float,
        tempo: Optional[float] = None,
        style_tags: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
        **kwargs
    ) -> Result[ProcessingResult]:
        """Generate beat using MusicGen"""

        if not self._model_loaded:
            return Result.err("Model not loaded. Call load_model() first.")

        if duration > self.MAX_DURATION:
            return Result.err(f"Duration {duration}s exceeds maximum {self.MAX_DURATION}s")

        try:
            # Enhance prompt with musical context
            enhanced_prompt = self._enhance_prompt(prompt, tempo, style_tags)

            # Process prompt
            inputs = self._processor(
                text=[enhanced_prompt],
                padding=True,
                return_tensors="pt"
            ).to(self.device)

            # Calculate generation parameters
            tokens_per_second = self.SAMPLE_RATE / self._model.config.audio_encoder.hop_length
            max_new_tokens = int(duration * tokens_per_second)

            # Generate with optimized settings
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "guidance_scale": guidance_scale,
                "temperature": temperature,
                "use_cache": True
            }

            start_time = time.time()

            audio_logger.log_processing_start(
                operation="MusicGenGeneration",
                prompt=enhanced_prompt[:100],
                duration=duration,
                model=self.current_model
            )

            # Generate in executor to avoid blocking
            loop = asyncio.get_event_loop()
            audio_values = await loop.run_in_executor(
                None,
                lambda: self._model.generate(**inputs, **generation_kwargs)
            )

            generation_time = time.time() - start_time

            # Convert to numpy
            audio_array = audio_values[0].cpu().numpy().squeeze()

            # Ensure correct length
            expected_samples = int(duration * self.SAMPLE_RATE)
            if len(audio_array) > expected_samples:
                audio_array = audio_array[:expected_samples]
            elif len(audio_array) < expected_samples:
                # Pad with zeros
                padding = expected_samples - len(audio_array)
                audio_array = np.pad(audio_array, (0, padding), mode='constant')

            # Quality assessment
            quality_score = self._assess_quality(audio_array, tempo)

            audio_logger.log_processing_complete(
                operation="MusicGenGeneration",
                duration_ms=generation_time * 1000,
                audio_duration=duration,
                quality_score=quality_score
            )

            return Result.ok(ProcessingResult(
                success=True,
                data=audio_array,
                metadata={
                    "sample_rate": self.SAMPLE_RATE,
                    "duration": duration,
                    "model": self.current_model,
                    "enhanced_prompt": enhanced_prompt,
                    "quality_score": quality_score,
                    "generation_time": generation_time,
                    "guidance_scale": guidance_scale,
                    "temperature": temperature
                }
            ))

        except torch.cuda.OutOfMemoryError:
            audio_logger.log_processing_error(
                operation="MusicGenOOM",
                error="GPU out of memory during generation"
            )

            if self.device == "cuda":
                # Try CPU fallback
                return Result.err(
                    "GPU out of memory. Try using a smaller model or reduce duration."
                )
            else:
                return Result.err("Insufficient memory for generation")

        except Exception as e:
            audio_logger.log_processing_error(
                operation="MusicGenGeneration",
                error=str(e)
            )
            return Result.err(f"MusicGen generation failed: {e}")

    def _enhance_prompt(
        self,
        prompt: str,
        tempo: Optional[float] = None,
        style_tags: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enhance prompt with musical context"""

        enhanced = prompt.strip()

        # Add tempo descriptor
        if tempo:
            if tempo < 70:
                enhanced += ", slow tempo"
            elif tempo < 90:
                enhanced += ", medium slow tempo"
            elif tempo < 110:
                enhanced += ", medium tempo"
            elif tempo < 130:
                enhanced += ", medium fast tempo"
            else:
                enhanced += ", fast tempo"

        # Add style information
        if style_tags:
            genre = style_tags.get("genre")
            if genre:
                enhanced += f", {genre} style"

            mood = style_tags.get("mood")
            if mood:
                enhanced += f", {mood} feeling"

            energy = style_tags.get("energy")
            if energy:
                enhanced += f", {energy} energy"

            instruments = style_tags.get("instruments")
            if instruments and isinstance(instruments, list):
                enhanced += f", featuring {', '.join(instruments[:3])}"

        return enhanced

    def _assess_quality(self, audio: np.ndarray, tempo: Optional[float] = None) -> float:
        """Assess generated audio quality"""

        try:
            if len(audio) == 0:
                return 0.0

            # Dynamic range assessment
            max_val = np.max(np.abs(audio))
            if max_val == 0:
                return 0.0

            # Normalize for analysis
            normalized = audio / max_val

            # RMS energy
            rms = np.sqrt(np.mean(normalized**2))
            energy_score = min(rms * 10, 5.0)

            # Spectral characteristics
            fft = np.fft.rfft(normalized)
            spectral_energy = np.sum(np.abs(fft)**2)
            spectral_score = min(np.log10(spectral_energy + 1) * 2, 3.0)

            # Dynamic range
            peak_to_rms = 20 * np.log10(max_val / (rms + 1e-8))
            dynamic_score = min(peak_to_rms / 10, 2.0)

            total_score = energy_score + spectral_score + dynamic_score
            return round(min(total_score, 10.0), 2)

        except Exception:
            return 7.0  # Default quality score

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model"""
        return self.SUPPORTED_MODELS.get(model_name)

    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._model_loaded

    async def cleanup(self) -> None:
        """Clean up resources"""
        if self._model is not None:
            del self._model
            self._model = None

        if self._processor is not None:
            del self._processor
            self._processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._model_loaded = False
        self.current_model = None