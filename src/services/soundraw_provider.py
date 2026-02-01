"""
SOUNDRAW Provider for AudioLab Beat Generation
API-based beat generation using SOUNDRAW's commercial AI music service
"""

import asyncio
import base64
import json
import numpy as np
from typing import Dict, Optional, Any, List
from pathlib import Path
import io

try:
    import httpx
    import soundfile as sf
    SOUNDRAW_AVAILABLE = True
except ImportError:
    SOUNDRAW_AVAILABLE = False

from ..core.audio_processor import ProcessingResult
from ..core.logging import audio_logger
from ..core.result import Result


class SOUNDRAWProvider:
    """SOUNDRAW API provider for beat generation"""

    BASE_URL = "https://api.soundraw.io/v1"
    MAX_DURATION = 300.0  # 5 minutes
    DEFAULT_SAMPLE_RATE = 44100

    # SOUNDRAW-specific style mappings
    GENRE_MAPPING = {
        "hip-hop": "hiphop",
        "hip hop": "hiphop",
        "electronic": "electronic",
        "edm": "electronic",
        "rock": "rock",
        "pop": "pop",
        "jazz": "jazz",
        "classical": "classical",
        "ambient": "ambient",
        "techno": "techno",
        "house": "house",
        "trap": "trap",
        "lo-fi": "lofi",
        "r&b": "rnb",
        "funk": "funk"
    }

    MOOD_MAPPING = {
        "energetic": "high_energy",
        "calm": "calm",
        "aggressive": "aggressive",
        "happy": "upbeat",
        "sad": "melancholic",
        "dark": "dark",
        "bright": "bright",
        "mysterious": "mysterious",
        "relaxing": "chill"
    }

    def __init__(self, api_key: str, timeout: float = 300.0):
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._initialized = False

    async def initialize(self) -> Result[None]:
        """Initialize SOUNDRAW API client"""

        if not SOUNDRAW_AVAILABLE:
            return Result.err(
                "SOUNDRAW dependencies not available. Install with: "
                "pip install httpx soundfile"
            )

        if not self.api_key:
            return Result.err("SOUNDRAW API key is required")

        try:
            self._client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "AudioLab/1.0"
                }
            )

            # Test API connection
            response = await self._client.get("/health")

            if response.status_code == 200:
                self._initialized = True

                audio_logger.log_processing_complete(
                    operation="SOUNDRAWInit",
                    duration_ms=0
                )

                return Result.ok(None)
            else:
                return Result.err(f"SOUNDRAW API health check failed: {response.status_code}")

        except httpx.RequestError as e:
            audio_logger.log_processing_error(
                operation="SOUNDRAWInit",
                error=str(e)
            )
            return Result.err(f"Failed to initialize SOUNDRAW client: {e}")

    async def generate(
        self,
        prompt: str,
        duration: float,
        tempo: Optional[float] = None,
        time_signature: Optional[str] = None,
        style_tags: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Result[ProcessingResult]:
        """Generate beat using SOUNDRAW API"""

        if not self._initialized or not self._client:
            return Result.err("SOUNDRAW client not initialized. Call initialize() first.")

        if duration > self.MAX_DURATION:
            return Result.err(f"Duration {duration}s exceeds maximum {self.MAX_DURATION}s")

        try:
            # Prepare generation request
            request_data = self._prepare_request(
                prompt, duration, tempo, time_signature, style_tags
            )

            audio_logger.log_processing_start(
                operation="SOUNDRAWGeneration",
                prompt=prompt[:100],
                duration=duration,
                tempo=tempo
            )

            # Submit generation request
            response = await self._client.post("/generate", json=request_data)

            if response.status_code != 200:
                error_msg = f"SOUNDRAW API error: {response.status_code}"
                try:
                    error_data = response.json()
                    if "message" in error_data:
                        error_msg += f" - {error_data['message']}"
                except:
                    error_msg += f" - {response.text}"

                return Result.err(error_msg)

            response_data = response.json()

            # Handle different response formats
            if "generation_id" in response_data:
                # Async generation - poll for completion
                audio_data = await self._poll_completion(response_data["generation_id"])
            elif "audio_url" in response_data:
                # Direct URL response
                audio_data = await self._download_audio(response_data["audio_url"])
            elif "audio_data" in response_data:
                # Base64 encoded audio
                audio_data = self._decode_audio(response_data["audio_data"])
            else:
                return Result.err("Invalid SOUNDRAW API response format")

            if audio_data is None:
                return Result.err("No audio data received from SOUNDRAW")

            # Process audio data
            audio_array = await self._process_audio(audio_data)

            # Quality assessment
            quality_score = self._assess_quality(audio_array, tempo)

            audio_logger.log_processing_complete(
                operation="SOUNDRAWGeneration",
                duration_ms=0,  # API handles timing
                audio_duration=duration,
                quality_score=quality_score
            )

            return Result.ok(ProcessingResult(
                success=True,
                data=audio_array,
                metadata={
                    "sample_rate": self.DEFAULT_SAMPLE_RATE,
                    "duration": duration,
                    "provider": "soundraw",
                    "quality_score": quality_score,
                    "api_response": {k: v for k, v in response_data.items() if k != "audio_data"}
                }
            ))

        except httpx.RequestError as e:
            audio_logger.log_processing_error(
                operation="SOUNDRAWGeneration",
                error=f"Request error: {e}"
            )
            return Result.err(f"SOUNDRAW API request failed: {e}")

        except Exception as e:
            audio_logger.log_processing_error(
                operation="SOUNDRAWGeneration",
                error=str(e)
            )
            return Result.err(f"SOUNDRAW generation failed: {e}")

    def _prepare_request(
        self,
        prompt: str,
        duration: float,
        tempo: Optional[float] = None,
        time_signature: Optional[str] = None,
        style_tags: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare SOUNDRAW API request"""

        request = {
            "description": prompt,
            "duration": int(duration),
            "format": "wav",
            "sample_rate": self.DEFAULT_SAMPLE_RATE
        }

        # Add tempo if provided
        if tempo:
            request["tempo"] = int(tempo)

        # Add time signature if provided
        if time_signature:
            request["time_signature"] = time_signature

        # Process style tags
        if style_tags:
            # Map genre
            genre = style_tags.get("genre", "").lower()
            mapped_genre = self.GENRE_MAPPING.get(genre)
            if mapped_genre:
                request["genre"] = mapped_genre

            # Map mood
            mood = style_tags.get("mood", "").lower()
            mapped_mood = self.MOOD_MAPPING.get(mood)
            if mapped_mood:
                request["mood"] = mapped_mood

            # Add energy level
            energy = style_tags.get("energy")
            if energy:
                if energy.lower() in ["low", "medium", "high"]:
                    request["energy"] = energy.lower()

            # Add instruments if supported
            instruments = style_tags.get("instruments")
            if instruments and isinstance(instruments, list):
                # SOUNDRAW may support instrument preferences
                request["instruments"] = instruments[:5]  # Limit to 5

        return request

    async def _poll_completion(self, generation_id: str) -> Optional[bytes]:
        """Poll SOUNDRAW API for generation completion"""

        max_attempts = 60  # 5 minutes max
        poll_interval = 5  # seconds

        for attempt in range(max_attempts):
            try:
                response = await self._client.get(f"/status/{generation_id}")

                if response.status_code == 200:
                    data = response.json()
                    status = data.get("status", "").lower()

                    if status == "completed":
                        if "audio_url" in data:
                            return await self._download_audio(data["audio_url"])
                        elif "audio_data" in data:
                            return self._decode_audio(data["audio_data"])
                    elif status == "failed":
                        error = data.get("error", "Unknown error")
                        raise Exception(f"SOUNDRAW generation failed: {error}")
                    elif status in ["pending", "processing"]:
                        # Continue polling
                        await asyncio.sleep(poll_interval)
                        continue
                    else:
                        raise Exception(f"Unknown SOUNDRAW status: {status}")

                else:
                    # API error during polling
                    raise Exception(f"SOUNDRAW polling error: {response.status_code}")

            except httpx.RequestError as e:
                # Network error during polling - continue trying
                await asyncio.sleep(poll_interval)
                continue

        raise Exception("SOUNDRAW generation timeout - exceeded maximum wait time")

    async def _download_audio(self, url: str) -> bytes:
        """Download audio from SOUNDRAW URL"""

        response = await self._client.get(url)

        if response.status_code == 200:
            return response.content
        else:
            raise Exception(f"Failed to download audio: {response.status_code}")

    def _decode_audio(self, audio_data: str) -> bytes:
        """Decode base64 audio data"""

        try:
            return base64.b64decode(audio_data)
        except Exception as e:
            raise Exception(f"Failed to decode audio data: {e}")

    async def _process_audio(self, audio_data: bytes) -> np.ndarray:
        """Process audio bytes to numpy array"""

        try:
            # Create file-like object from bytes
            audio_buffer = io.BytesIO(audio_data)

            # Read audio using soundfile in executor
            loop = asyncio.get_event_loop()
            audio_array, sample_rate = await loop.run_in_executor(
                None,
                lambda: sf.read(audio_buffer)
            )

            # Resample if needed
            if sample_rate != self.DEFAULT_SAMPLE_RATE:
                # Use simple resampling for now
                # In production, would use librosa.resample
                ratio = self.DEFAULT_SAMPLE_RATE / sample_rate
                new_length = int(len(audio_array) * ratio)
                audio_array = np.interp(
                    np.linspace(0, len(audio_array) - 1, new_length),
                    np.arange(len(audio_array)),
                    audio_array
                )

            # Ensure float32 format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)

            return audio_array

        except Exception as e:
            raise Exception(f"Failed to process audio: {e}")

    def _assess_quality(self, audio: np.ndarray, tempo: Optional[float] = None) -> float:
        """Assess generated audio quality"""

        try:
            if len(audio) == 0:
                return 0.0

            # Basic quality metrics
            max_val = np.max(np.abs(audio))
            if max_val == 0:
                return 0.0

            # RMS energy
            rms = np.sqrt(np.mean(audio**2))
            energy_score = min(rms * 8, 4.0)

            # Dynamic range
            dynamic_range = np.max(audio) - np.min(audio)
            dynamic_score = min(dynamic_range * 3, 3.0)

            # Frequency content
            fft = np.fft.rfft(audio)
            spectral_centroid = np.sum(np.arange(len(fft)) * np.abs(fft)) / (np.sum(np.abs(fft)) + 1e-8)
            spectral_score = min(spectral_centroid / 1000, 3.0)

            total_score = energy_score + dynamic_score + spectral_score
            return round(min(total_score, 10.0), 2)

        except Exception:
            return 8.0  # Higher default for commercial API

    async def get_credits_remaining(self) -> Result[int]:
        """Get remaining API credits"""

        try:
            response = await self._client.get("/account/credits")

            if response.status_code == 200:
                data = response.json()
                return Result.ok(data.get("credits_remaining", 0))
            else:
                return Result.err(f"Failed to get credits: {response.status_code}")

        except Exception as e:
            return Result.err(f"Credits check failed: {e}")

    async def cleanup(self) -> None:
        """Clean up resources"""

        if self._client:
            await self._client.aclose()
            self._client = None

        self._initialized = False