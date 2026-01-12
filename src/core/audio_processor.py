"""
AudioLab Core Audio Processing Engine
Implementation of audio processing patterns from .claude/patterns/AUDIO_PATTERNS.md
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from pathlib import Path
import threading
from collections import deque

import numpy as np
from pydantic import BaseModel
from enum import Enum
import soundfile as sf
from scipy import signal

from ..core.logging import audio_logger, performance_logger


class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    FLAC = "flac"
    MP3 = "mp3"
    AIFF = "aiff"


class ProcessingResult(BaseModel):
    """Result of audio processing operation"""
    success: bool
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0


class AudioProcessor(Protocol):
    """Protocol for all audio processing operations"""
    async def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        **kwargs
    ) -> ProcessingResult:
        ...


class BaseAudioProcessor(ABC):
    """Base class for audio processors with common functionality"""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._is_processing = False

    @abstractmethod
    async def _process_internal(
        self,
        audio: np.ndarray,
        **kwargs
    ) -> ProcessingResult:
        """Internal processing implementation"""
        pass

    async def process(
        self,
        audio: np.ndarray,
        **kwargs
    ) -> ProcessingResult:
        """Public processing interface with error handling"""
        if self._is_processing:
            return ProcessingResult(
                success=False,
                error="Processor already running"
            )

        try:
            self._is_processing = True
            start_time = time.time()

            # Validate input
            if not self._validate_audio(audio):
                return ProcessingResult(
                    success=False,
                    error="Invalid audio input"
                )

            result = await self._process_internal(audio, **kwargs)
            result.processing_time = time.time() - start_time

            # Log processing completion
            audio_logger.log_processing_complete(
                operation=self.__class__.__name__,
                duration_ms=result.processing_time * 1000,
                **kwargs
            )

            return result

        except Exception as e:
            audio_logger.log_processing_error(
                operation=self.__class__.__name__,
                error=str(e),
                **kwargs
            )
            return ProcessingResult(
                success=False,
                error=f"Processing failed: {str(e)}"
            )
        finally:
            self._is_processing = False

    def _validate_audio(self, audio: np.ndarray) -> bool:
        """Validate audio array format"""
        if audio is None or audio.size == 0:
            return False
        if audio.dtype not in [np.float32, np.float64]:
            return False
        if len(audio.shape) > 2:  # Mono or stereo only
            return False
        return True


class ParametricEQ(BaseAudioProcessor):
    """Parametric EQ with multiple bands"""

    def __init__(self, sample_rate: int = 48000):
        super().__init__(sample_rate)
        self.bands: List[Dict[str, float]] = []

    def add_band(
        self,
        freq: float,
        gain_db: float,
        q: float = 1.0,
        band_type: str = "bell"
    ) -> None:
        """Add EQ band"""
        self.bands.append({
            "freq": freq,
            "gain": gain_db,
            "q": q,
            "type": band_type
        })

    async def _process_internal(
        self,
        audio: np.ndarray,
        **kwargs
    ) -> ProcessingResult:
        """Apply EQ processing"""
        try:
            processed = audio.copy()

            for band in self.bands:
                if band["type"] == "highpass":
                    sos = signal.butter(
                        2, band["freq"],
                        btype='high',
                        fs=self.sample_rate,
                        output='sos'
                    )
                elif band["type"] == "lowpass":
                    sos = signal.butter(
                        2, band["freq"],
                        btype='low',
                        fs=self.sample_rate,
                        output='sos'
                    )
                elif band["type"] == "bell":
                    # Bell filter implementation
                    w0 = 2 * np.pi * band["freq"] / self.sample_rate
                    A = 10 ** (band["gain"] / 40)
                    alpha = np.sin(w0) / (2 * band["q"])

                    # Biquad coefficients
                    b0 = 1 + alpha * A
                    b1 = -2 * np.cos(w0)
                    b2 = 1 - alpha * A
                    a0 = 1 + alpha / A
                    a1 = -2 * np.cos(w0)
                    a2 = 1 - alpha / A

                    # Normalize
                    sos = np.array([[[b0/a0, b1/a0, b2/a0, 1, a1/a0, a2/a0]]])

                # Apply filter
                if len(processed.shape) == 1:
                    processed = signal.sosfilt(sos, processed)
                else:
                    for ch in range(processed.shape[1]):
                        processed[:, ch] = signal.sosfilt(sos, processed[:, ch])

            return ProcessingResult(
                success=True,
                data=processed,
                metadata={"bands_applied": len(self.bands)}
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"EQ processing failed: {e}"
            )


class Compressor(BaseAudioProcessor):
    """Dynamic range compressor"""

    def __init__(
        self,
        sample_rate: int = 48000,
        threshold: float = -12.0,  # dB
        ratio: float = 4.0,
        attack: float = 10.0,      # ms
        release: float = 100.0     # ms
    ):
        super().__init__(sample_rate)
        self.threshold = threshold
        self.ratio = ratio
        self.attack = attack
        self.release = release

        # Calculate time constants
        self.attack_coeff = np.exp(-1.0 / (attack * 0.001 * sample_rate))
        self.release_coeff = np.exp(-1.0 / (release * 0.001 * sample_rate))

        self.envelope = 0.0

    async def _process_internal(
        self,
        audio: np.ndarray,
        **kwargs
    ) -> ProcessingResult:
        """Apply compression"""
        try:
            processed = audio.copy()

            if len(processed.shape) == 1:
                processed = processed[:, np.newaxis]

            for i in range(len(processed)):
                # Get peak level
                sample_level = np.max(np.abs(processed[i, :]))

                # Convert to dB
                if sample_level > 0:
                    level_db = 20 * np.log10(sample_level)
                else:
                    level_db = -np.inf

                # Update envelope
                if level_db > self.envelope:
                    self.envelope = self.attack_coeff * self.envelope + \
                                   (1 - self.attack_coeff) * level_db
                else:
                    self.envelope = self.release_coeff * self.envelope + \
                                   (1 - self.release_coeff) * level_db

                # Calculate gain reduction
                if self.envelope > self.threshold:
                    gain_reduction = (self.envelope - self.threshold) * \
                                   (1 - 1/self.ratio)
                    gain_linear = 10 ** (-gain_reduction / 20)
                else:
                    gain_linear = 1.0

                # Apply gain
                processed[i, :] *= gain_linear

            if audio.shape != processed.shape:
                processed = processed.squeeze()

            return ProcessingResult(
                success=True,
                data=processed,
                metadata={
                    "threshold": self.threshold,
                    "ratio": self.ratio
                }
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Compression failed: {e}"
            )


class AudioBuffer:
    """Thread-safe circular buffer for real-time audio"""

    def __init__(
        self,
        buffer_size: int = 4096,
        channels: int = 2,
        sample_rate: int = 48000
    ):
        self.buffer_size = buffer_size
        self.channels = channels
        self.sample_rate = sample_rate

        # Thread-safe buffer
        self._buffer = deque(maxlen=buffer_size)
        self._lock = threading.Lock()

        # Statistics
        self.underruns = 0
        self.overruns = 0

    def write(self, audio_chunk: np.ndarray) -> bool:
        """Write audio chunk to buffer"""
        with self._lock:
            if len(self._buffer) >= self.buffer_size:
                self.overruns += 1
                return False

            self._buffer.append(audio_chunk.copy())
            return True

    def read(self, num_samples: int) -> Optional[np.ndarray]:
        """Read audio chunk from buffer"""
        with self._lock:
            if len(self._buffer) == 0:
                self.underruns += 1
                return None

            chunk = self._buffer.popleft()
            return chunk

    def available(self) -> int:
        """Get number of available chunks"""
        with self._lock:
            return len(self._buffer)


class RealTimeAudioProcessor:
    """Real-time audio processing with buffering"""

    def __init__(
        self,
        buffer_size: int = 512,
        sample_rate: int = 48000,
        channels: int = 2
    ):
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate
        self.channels = channels

        self.input_buffer = AudioBuffer(buffer_size, channels, sample_rate)
        self.output_buffer = AudioBuffer(buffer_size, channels, sample_rate)

        self.effects_chain: List[AudioProcessor] = []
        self.is_processing = False
        self._processing_thread = None

    def add_effect(self, effect: AudioProcessor) -> None:
        """Add effect to processing chain"""
        self.effects_chain.append(effect)

    def start_processing(self) -> None:
        """Start real-time processing thread"""
        if self.is_processing:
            return

        self.is_processing = True
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self._processing_thread.start()

        audio_logger.log_processing_start(
            operation="RealTimeProcessing",
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size
        )

    def stop_processing(self) -> None:
        """Stop real-time processing"""
        self.is_processing = False
        if self._processing_thread:
            self._processing_thread.join(timeout=1.0)

    def _processing_loop(self) -> None:
        """Main real-time processing loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while self.is_processing:
            # Read from input buffer
            audio_chunk = self.input_buffer.read(self.buffer_size)
            if audio_chunk is None:
                continue

            # Apply effects chain
            processed = audio_chunk
            for effect in self.effects_chain:
                try:
                    # Run async processing in sync context
                    result = loop.run_until_complete(
                        effect.process(processed)
                    )

                    if result.success:
                        processed = result.data

                except Exception as e:
                    # Skip effect on error to maintain real-time
                    audio_logger.log_processing_error(
                        operation="RealTimeEffect",
                        error=str(e)
                    )
                    continue

            # Write to output buffer
            self.output_buffer.write(processed)

            # Log real-time stats periodically
            if hasattr(self, '_last_stats_log'):
                if time.time() - self._last_stats_log > 5.0:  # Every 5 seconds
                    self._log_realtime_stats()
            else:
                self._last_stats_log = time.time()

        loop.close()

    def _log_realtime_stats(self) -> None:
        """Log real-time processing statistics"""
        audio_logger.log_realtime_stats(
            buffer_underruns=self.input_buffer.underruns,
            buffer_overruns=self.input_buffer.overruns,
            latency_ms=(self.buffer_size / self.sample_rate) * 1000,
            cpu_usage=0.0  # TODO: Implement CPU monitoring
        )
        self._last_stats_log = time.time()


class AudioFileManager:
    """High-performance audio file operations"""

    SUPPORTED_FORMATS = {
        ".wav": "wav",
        ".flac": "flac",
        ".mp3": "mp3",
        ".aif": "aiff",
        ".aiff": "aiff"
    }

    @staticmethod
    async def load_audio(
        file_path: str,
        target_sr: Optional[int] = None,
        mono: bool = False
    ) -> ProcessingResult:
        """Load audio file with format validation"""
        try:
            path = Path(file_path)

            if not path.exists():
                return ProcessingResult(
                    success=False,
                    error=f"File not found: {file_path}"
                )

            if path.suffix.lower() not in AudioFileManager.SUPPORTED_FORMATS:
                return ProcessingResult(
                    success=False,
                    error=f"Unsupported format: {path.suffix}"
                )

            # Load in thread to avoid blocking
            audio, sr = await asyncio.to_thread(sf.read, file_path)

            # Convert to mono if requested
            if mono and len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if target_sr and sr != target_sr:
                import librosa
                audio = await asyncio.to_thread(
                    librosa.resample,
                    audio,
                    orig_sr=sr,
                    target_sr=target_sr
                )
                sr = target_sr

            audio_logger.log_processing_complete(
                operation="AudioLoad",
                duration_ms=0,  # Instantaneous
                file_path=file_path
            )

            return ProcessingResult(
                success=True,
                data=audio,
                metadata={
                    "sample_rate": sr,
                    "duration": len(audio) / sr,
                    "channels": 1 if len(audio.shape) == 1 else audio.shape[1],
                    "format": path.suffix
                }
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Failed to load audio: {e}"
            )

    @staticmethod
    async def save_audio(
        audio: np.ndarray,
        file_path: str,
        sample_rate: int = 48000,
        bit_depth: int = 24
    ) -> ProcessingResult:
        """Save audio file with quality settings"""
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            # Convert bit depth
            if bit_depth == 16:
                subtype = 'PCM_16'
            elif bit_depth == 24:
                subtype = 'PCM_24'
            elif bit_depth == 32:
                subtype = 'PCM_32'
            else:
                subtype = 'PCM_24'  # Default

            # Save in thread
            await asyncio.to_thread(
                sf.write,
                file_path,
                audio,
                sample_rate,
                subtype=subtype
            )

            audio_logger.log_processing_complete(
                operation="AudioSave",
                duration_ms=0,  # Instantaneous
                file_path=file_path
            )

            return ProcessingResult(
                success=True,
                metadata={
                    "file_path": file_path,
                    "sample_rate": sample_rate,
                    "bit_depth": bit_depth,
                    "file_size": path.stat().st_size
                }
            )

        except Exception as e:
            return ProcessingResult(
                success=False,
                error=f"Failed to save audio: {e}"
            )