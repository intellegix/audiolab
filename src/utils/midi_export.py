"""
MIDI Export Utilities for AudioLab Beat Generation
Convert generated beats to MIDI using onset detection and rhythmic analysis
"""

import asyncio
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import tempfile
from decimal import Decimal

try:
    import pretty_midi
    import librosa
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

from ..core.audio_processor import AudioFileManager, ProcessingResult
from ..core.logging import audio_logger
from ..core.result import Result


class MidiExportService:
    """Service for converting audio beats to MIDI files"""

    # Standard drum mapping (General MIDI Level 1)
    DRUM_MAP = {
        "kick": 36,      # Bass Drum 1
        "snare": 38,     # Acoustic Snare
        "hihat": 42,     # Closed Hi Hat
        "open_hihat": 46, # Open Hi Hat
        "crash": 49,     # Crash Cymbal 1
        "ride": 51,      # Ride Cymbal 1
        "tom_low": 43,   # Low Floor Tom
        "tom_mid": 47,   # Low-Mid Tom
        "tom_high": 50,  # High Tom
        "clap": 39,      # Hand Clap
        "cowbell": 56,   # Cowbell
        "tambourine": 54  # Tambourine
    }

    def __init__(self):
        self.sample_rate = 44100
        self.hop_length = 512

    async def convert_beat_to_midi(
        self,
        audio_path: str,
        output_path: str,
        tempo: float,
        time_signature: str = "4/4",
        quantize: bool = True,
        sensitivity: float = 0.3
    ) -> Result[ProcessingResult]:
        """Convert beat audio to MIDI file using onset detection"""

        if not MIDI_AVAILABLE:
            return Result.err(
                "MIDI export dependencies not available. Install with: "
                "pip install pretty_midi librosa"
            )

        try:
            audio_logger.log_processing_start(
                operation="BeatToMIDI",
                audio_path=audio_path,
                tempo=tempo
            )

            # Load audio
            audio_result = await AudioFileManager.load_audio(audio_path)
            if not audio_result.success:
                return Result.err(f"Failed to load audio: {audio_result.error}")

            audio = audio_result.data
            source_sr = audio_result.metadata["sample_rate"]

            # Resample if needed
            if source_sr != self.sample_rate:
                audio = await asyncio.to_thread(
                    librosa.resample,
                    audio, orig_sr=source_sr, target_sr=self.sample_rate
                )

            # Detect onsets
            onsets = await self._detect_onsets(audio, sensitivity)

            # Analyze rhythm and extract drum patterns
            drum_patterns = await self._analyze_drum_patterns(audio, onsets)

            # Create MIDI from patterns
            midi_data = await self._create_midi_from_patterns(
                drum_patterns, tempo, time_signature, quantize
            )

            # Save MIDI file
            save_result = await self._save_midi_file(midi_data, output_path)
            if not save_result.success:
                return Result.err(f"Failed to save MIDI: {save_result.error}")

            audio_logger.log_processing_complete(
                operation="BeatToMIDI",
                duration_ms=0,
                output_path=output_path,
                onset_count=len(onsets),
                pattern_count=len(drum_patterns)
            )

            return Result.ok(ProcessingResult(
                success=True,
                data=output_path,
                metadata={
                    "tempo": tempo,
                    "time_signature": time_signature,
                    "onset_count": len(onsets),
                    "drum_patterns": list(drum_patterns.keys()),
                    "quantized": quantize,
                    "duration": len(audio) / self.sample_rate
                }
            ))

        except Exception as e:
            audio_logger.log_processing_error(
                operation="BeatToMIDI",
                error=str(e)
            )
            return Result.err(f"MIDI conversion failed: {e}")

    async def _detect_onsets(self, audio: np.ndarray, sensitivity: float) -> np.ndarray:
        """Detect onset times in audio"""

        try:
            # Use librosa onset detection
            onset_frames = await asyncio.to_thread(
                librosa.onset.onset_detect,
                y=audio,
                sr=self.sample_rate,
                hop_length=self.hop_length,
                units='time',
                delta=sensitivity
            )

            return onset_frames

        except Exception as e:
            # Fallback to simple energy-based detection
            return await self._simple_onset_detection(audio)

    async def _simple_onset_detection(self, audio: np.ndarray) -> np.ndarray:
        """Simple energy-based onset detection as fallback"""

        # Calculate RMS energy in frames
        frame_size = self.hop_length
        frames = []

        for i in range(0, len(audio) - frame_size, frame_size):
            frame = audio[i:i + frame_size]
            rms = np.sqrt(np.mean(frame**2))
            frames.append(rms)

        frames = np.array(frames)

        # Find peaks in energy
        threshold = np.mean(frames) + 2 * np.std(frames)
        peaks = []

        for i in range(1, len(frames) - 1):
            if (frames[i] > frames[i-1] and
                frames[i] > frames[i+1] and
                frames[i] > threshold):
                # Convert frame index to time
                time = (i * frame_size) / self.sample_rate
                peaks.append(time)

        return np.array(peaks)

    async def _analyze_drum_patterns(
        self,
        audio: np.ndarray,
        onsets: np.ndarray
    ) -> Dict[str, List[float]]:
        """Analyze audio to identify different drum sounds"""

        patterns = {
            "kick": [],
            "snare": [],
            "hihat": []
        }

        try:
            # Simple frequency-based classification
            for onset_time in onsets:
                onset_sample = int(onset_time * self.sample_rate)

                # Extract a small window around onset
                window_size = int(0.1 * self.sample_rate)  # 100ms
                start = max(0, onset_sample - window_size // 2)
                end = min(len(audio), onset_sample + window_size // 2)
                window = audio[start:end]

                if len(window) == 0:
                    continue

                # Analyze frequency content
                fft = np.fft.rfft(window)
                freqs = np.fft.rfftfreq(len(window), 1/self.sample_rate)
                magnitude = np.abs(fft)

                # Classify based on dominant frequency
                dominant_freq_idx = np.argmax(magnitude)
                dominant_freq = freqs[dominant_freq_idx]

                # Simple classification rules
                if dominant_freq < 100:
                    patterns["kick"].append(onset_time)
                elif dominant_freq > 5000:
                    patterns["hihat"].append(onset_time)
                else:
                    patterns["snare"].append(onset_time)

            return patterns

        except Exception as e:
            # Fallback: distribute onsets evenly across drum types
            return {
                "kick": onsets[::3].tolist(),
                "snare": onsets[1::3].tolist(),
                "hihat": onsets[2::3].tolist()
            }

    async def _create_midi_from_patterns(
        self,
        patterns: Dict[str, List[float]],
        tempo: float,
        time_signature: str,
        quantize: bool
    ) -> pretty_midi.PrettyMIDI:
        """Create MIDI data from drum patterns"""

        # Create MIDI object
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Create drum track (channel 9)
        drum_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        drums = pretty_midi.Instrument(program=drum_program, is_drum=True)

        # Parse time signature
        numerator, denominator = map(int, time_signature.split('/'))
        beats_per_bar = numerator
        beat_duration = 60.0 / tempo  # seconds per beat

        # Add notes for each drum pattern
        for drum_type, times in patterns.items():
            if drum_type not in self.DRUM_MAP:
                continue

            note_number = self.DRUM_MAP[drum_type]

            for time in times:
                # Quantize to nearest beat if requested
                if quantize:
                    beat_number = round(time / beat_duration)
                    quantized_time = beat_number * beat_duration
                else:
                    quantized_time = time

                # Create note (short duration for drum hits)
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=note_number,
                    start=quantized_time,
                    end=quantized_time + 0.1  # 100ms duration
                )
                drums.notes.append(note)

        # Sort notes by time
        drums.notes.sort(key=lambda n: n.start)

        # Add drum track to MIDI
        midi.instruments.append(drums)

        return midi

    async def _save_midi_file(self, midi_data: pretty_midi.PrettyMIDI, output_path: str) -> Result[bool]:
        """Save MIDI data to file"""

        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save MIDI file
            await asyncio.to_thread(midi_data.write, output_path)

            return Result.ok(True)

        except Exception as e:
            return Result.err(f"Failed to save MIDI file: {e}")

    async def create_midi_template(
        self,
        tempo: float,
        time_signature: str,
        pattern_type: str = "basic_rock",
        bars: int = 4
    ) -> Result[str]:
        """Create a MIDI template with predefined drum patterns"""

        if not MIDI_AVAILABLE:
            return Result.err("MIDI dependencies not available")

        try:
            # Define pattern templates
            patterns = self._get_pattern_templates()

            if pattern_type not in patterns:
                return Result.err(f"Unknown pattern type: {pattern_type}")

            pattern = patterns[pattern_type]

            # Create MIDI
            midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
            drums = pretty_midi.Instrument(program=0, is_drum=True)

            # Parse time signature
            numerator, denominator = map(int, time_signature.split('/'))
            beats_per_bar = numerator
            beat_duration = 60.0 / tempo
            bar_duration = beats_per_bar * beat_duration

            # Generate pattern for specified number of bars
            for bar in range(bars):
                bar_start_time = bar * bar_duration

                for drum_type, beats in pattern.items():
                    if drum_type not in self.DRUM_MAP:
                        continue

                    note_number = self.DRUM_MAP[drum_type]

                    for beat in beats:
                        if beat < beats_per_bar:  # Ensure beat is within bar
                            note_time = bar_start_time + (beat * beat_duration)

                            note = pretty_midi.Note(
                                velocity=100,
                                pitch=note_number,
                                start=note_time,
                                end=note_time + 0.1
                            )
                            drums.notes.append(note)

            midi.instruments.append(drums)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                midi.write(tmp.name)
                return Result.ok(tmp.name)

        except Exception as e:
            return Result.err(f"Failed to create MIDI template: {e}")

    def _get_pattern_templates(self) -> Dict[str, Dict[str, List[float]]]:
        """Get predefined drum pattern templates"""

        return {
            "basic_rock": {
                "kick": [0, 2],  # Beats 1 and 3
                "snare": [1, 3],  # Beats 2 and 4
                "hihat": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]  # Eighth notes
            },
            "basic_pop": {
                "kick": [0, 1.5, 2],
                "snare": [1, 3],
                "hihat": [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
            },
            "latin": {
                "kick": [0, 1, 2, 3],  # Four on the floor
                "snare": [1.5, 3.5],
                "hihat": [0.5, 1, 2, 2.5]
            },
            "hip_hop": {
                "kick": [0, 0.75, 2, 2.75],
                "snare": [1, 3],
                "hihat": [0.5, 1.5, 2.5, 3.5]
            },
            "jazz": {
                "kick": [0, 2],
                "snare": [1, 3],
                "ride": [0, 0.333, 0.666, 1, 1.333, 1.666, 2, 2.333, 2.666, 3, 3.333, 3.666]  # Swing feel
            }
        }

    async def enhance_midi_with_velocity(
        self,
        midi_path: str,
        audio_path: str,
        output_path: Optional[str] = None
    ) -> Result[str]:
        """Enhance MIDI file with velocity data extracted from audio"""

        if not output_path:
            output_path = midi_path.replace('.mid', '_enhanced.mid')

        try:
            # Load MIDI and audio
            midi = pretty_midi.PrettyMIDI(midi_path)
            audio_result = await AudioFileManager.load_audio(audio_path)

            if not audio_result.success:
                return Result.err(f"Failed to load audio: {audio_result.error}")

            audio = audio_result.data

            # Extract velocity information from audio
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    continue

                for note in instrument.notes:
                    # Get audio energy around note time
                    note_sample = int(note.start * self.sample_rate)
                    window_size = int(0.05 * self.sample_rate)  # 50ms window

                    start = max(0, note_sample - window_size // 2)
                    end = min(len(audio), note_sample + window_size // 2)

                    if end > start:
                        window = audio[start:end]
                        energy = np.sqrt(np.mean(window**2))

                        # Map energy to MIDI velocity (1-127)
                        velocity = int(np.clip(energy * 500, 1, 127))
                        note.velocity = velocity

            # Save enhanced MIDI
            await asyncio.to_thread(midi.write, output_path)

            return Result.ok(output_path)

        except Exception as e:
            return Result.err(f"MIDI enhancement failed: {e}")

    def get_supported_patterns(self) -> List[str]:
        """Get list of supported pattern templates"""
        return list(self._get_pattern_templates().keys())

    def validate_midi_dependencies(self) -> Dict[str, bool]:
        """Validate MIDI export dependencies"""
        return {
            "pretty_midi": MIDI_AVAILABLE,
            "mido": MIDO_AVAILABLE,
            "librosa": True  # Already validated in other services
        }