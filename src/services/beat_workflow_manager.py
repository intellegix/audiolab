"""
Beat Workflow Manager for AudioLab
Orchestrates the complete beat generation to project integration workflow
"""

import uuid
import asyncio
from typing import Dict, List, Optional, Any, Callable
from decimal import Decimal
from pathlib import Path

from ..core.logging import audio_logger
from ..core.result import Result
from ..services.beat_generation_service import BeatGenerationService
from ..services.project_integration_service import ProjectIntegrationService
from ..utils.midi_export import MidiExportService
from ..database.repositories.beat_generation_repository import BeatGenerationRepository
from ..database.schemas import (
    BeatGenerationRequest,
    BeatGenerationUpdate,
    BeatVariationCreate
)


class BeatWorkflowManager:
    """Manages complete beat generation workflows from request to project integration"""

    def __init__(self):
        self.generation_service = BeatGenerationService()
        self.integration_service = ProjectIntegrationService()
        self.midi_service = MidiExportService()

    async def execute_complete_beat_workflow(
        self,
        request: BeatGenerationRequest,
        project_id: uuid.UUID,
        user_id: uuid.UUID,
        track_id: Optional[uuid.UUID] = None,
        auto_add_to_project: bool = True,
        generate_variations: int = 1,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Result[Dict[str, Any]]:
        """Execute complete workflow from generation to project integration"""

        workflow_id = uuid.uuid4()

        try:
            audio_logger.log_processing_start(
                operation="CompleteBeatWorkflow",
                workflow_id=str(workflow_id),
                project_id=str(project_id),
                provider=request.provider,
                variations=generate_variations
            )

            if progress_callback:
                progress_callback(0.1, "Initializing beat generation workflow")

            # Step 1: Generate primary beat
            if progress_callback:
                progress_callback(0.2, f"Generating beat with {request.provider}")

            generation_result = await self.generation_service.generate_beat(
                prompt=request.prompt,
                duration=float(request.duration),
                tempo=float(request.tempo),
                time_signature=request.time_signature,
                style_tags=request.style_tags,
                project_id=project_id
            )

            if not generation_result.success:
                return Result.err(f"Beat generation failed: {generation_result.error}")

            primary_beat_audio = generation_result.data
            primary_metadata = generation_result.metadata

            # Step 2: Generate variations if requested
            variations = []
            if generate_variations > 1:
                if progress_callback:
                    progress_callback(0.4, f"Generating {generate_variations - 1} variations")

                variations = await self._generate_beat_variations(
                    request,
                    generate_variations - 1,
                    progress_callback
                )

            # Step 3: Save all generated beats
            if progress_callback:
                progress_callback(0.6, "Saving generated beats")

            saved_beats = await self._save_generated_beats(
                workflow_id,
                primary_beat_audio,
                variations,
                primary_metadata
            )

            if not saved_beats.success:
                return Result.err(f"Failed to save beats: {saved_beats.error}")

            beat_files = saved_beats.data

            # Step 4: Export MIDI files
            if progress_callback:
                progress_callback(0.7, "Exporting MIDI files")

            midi_results = await self._export_midi_for_beats(
                beat_files,
                float(request.tempo),
                request.time_signature
            )

            # Step 5: Integrate into project if requested
            integration_results = None
            if auto_add_to_project:
                if progress_callback:
                    progress_callback(0.8, "Adding beats to project")

                integration_results = await self._integrate_beats_into_project(
                    project_id,
                    beat_files,
                    midi_results,
                    track_id,
                    request
                )

                if not integration_results.success:
                    # Don't fail entire workflow if integration fails
                    audio_logger.log_processing_error(
                        operation="BeatIntegrationWarning",
                        error=f"Integration failed: {integration_results.error}"
                    )

            # Step 6: Create workflow summary
            if progress_callback:
                progress_callback(0.95, "Finalizing workflow")

            workflow_summary = {
                "workflow_id": workflow_id,
                "primary_beat": {
                    "audio_path": beat_files[0]["audio_path"],
                    "midi_path": midi_results.get(beat_files[0]["audio_path"]) if midi_results else None,
                    "quality_score": primary_metadata.get("quality_score"),
                    "metadata": primary_metadata
                },
                "variations": [
                    {
                        "index": i + 1,
                        "audio_path": beat_file["audio_path"],
                        "midi_path": midi_results.get(beat_file["audio_path"]) if midi_results else None,
                        "quality_score": beat_file.get("quality_score"),
                        "metadata": beat_file.get("metadata", {})
                    }
                    for i, beat_file in enumerate(beat_files[1:])
                ],
                "project_integration": integration_results.data if integration_results and integration_results.success else None,
                "generation_parameters": {
                    "prompt": request.prompt,
                    "provider": request.provider,
                    "duration": request.duration,
                    "tempo": request.tempo,
                    "time_signature": request.time_signature,
                    "style_tags": request.style_tags
                },
                "workflow_stats": {
                    "total_beats_generated": len(beat_files),
                    "midi_files_exported": len(midi_results) if midi_results else 0,
                    "project_integrated": integration_results.success if integration_results else False
                }
            }

            if progress_callback:
                progress_callback(1.0, "Beat workflow completed successfully")

            audio_logger.log_processing_complete(
                operation="CompleteBeatWorkflow",
                duration_ms=5000,  # Approximate
                workflow_id=str(workflow_id),
                beats_generated=len(beat_files),
                project_integrated=integration_results.success if integration_results else False
            )

            return Result.ok(workflow_summary)

        except Exception as e:
            audio_logger.log_processing_error(
                operation="CompleteBeatWorkflow",
                error=str(e),
                workflow_id=str(workflow_id)
            )
            return Result.err(f"Beat workflow failed: {e}")

        finally:
            # Cleanup generation service
            await self.generation_service.cleanup()

    async def _generate_beat_variations(
        self,
        base_request: BeatGenerationRequest,
        variation_count: int,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[Dict[str, Any]]:
        """Generate beat variations with modified parameters"""

        variations = []

        # Variation strategies
        strategies = [
            {"name": "tempo_up", "tempo_mod": 1.1, "prompt_suffix": ", energetic"},
            {"name": "tempo_down", "tempo_mod": 0.9, "prompt_suffix": ", relaxed"},
            {"name": "style_variation", "style_mod": {"energy": "high"}, "prompt_suffix": ", intense"},
            {"name": "minimal", "prompt_prefix": "minimal ", "prompt_suffix": ", sparse"},
            {"name": "complex", "prompt_prefix": "complex ", "prompt_suffix": ", layered"}
        ]

        for i in range(variation_count):
            strategy = strategies[i % len(strategies)]

            if progress_callback:
                progress_callback(
                    0.4 + (0.2 * (i + 1) / variation_count),
                    f"Generating variation {i + 1}: {strategy['name']}"
                )

            # Modify request parameters
            variation_prompt = base_request.prompt
            if "prompt_prefix" in strategy:
                variation_prompt = strategy["prompt_prefix"] + variation_prompt
            if "prompt_suffix" in strategy:
                variation_prompt += strategy["prompt_suffix"]

            variation_tempo = base_request.tempo
            if "tempo_mod" in strategy:
                variation_tempo = float(base_request.tempo) * strategy["tempo_mod"]

            variation_style_tags = base_request.style_tags or {}
            if "style_mod" in strategy:
                variation_style_tags.update(strategy["style_mod"])

            # Generate variation
            try:
                generation_result = await self.generation_service.generate_beat(
                    prompt=variation_prompt,
                    duration=float(base_request.duration),
                    tempo=variation_tempo,
                    time_signature=base_request.time_signature,
                    style_tags=variation_style_tags,
                    project_id=base_request.project_id
                )

                if generation_result.success:
                    variations.append({
                        "strategy": strategy["name"],
                        "audio_data": generation_result.data,
                        "metadata": generation_result.metadata,
                        "quality_score": generation_result.metadata.get("quality_score", 7.0),
                        "variation_parameters": {
                            "prompt": variation_prompt,
                            "tempo": variation_tempo,
                            "style_tags": variation_style_tags
                        }
                    })

            except Exception as e:
                audio_logger.log_processing_error(
                    operation="VariationGeneration",
                    error=f"Variation {i + 1} failed: {e}"
                )
                continue

        return variations

    async def _save_generated_beats(
        self,
        workflow_id: uuid.UUID,
        primary_audio: Any,
        variations: List[Dict[str, Any]],
        primary_metadata: Dict[str, Any]
    ) -> Result[List[Dict[str, Any]]]:
        """Save all generated beats to files"""

        try:
            from ..core.config import get_settings
            settings = get_settings()

            beat_files = []

            # Save primary beat
            primary_filename = f"beat_{workflow_id}_primary.wav"
            primary_path = Path(settings.BEATS_PATH) / primary_filename

            # TODO: Implement actual audio saving
            # For now, create placeholder files
            primary_path.parent.mkdir(parents=True, exist_ok=True)
            primary_path.touch()

            beat_files.append({
                "type": "primary",
                "audio_path": str(primary_path),
                "metadata": primary_metadata,
                "quality_score": primary_metadata.get("quality_score", 7.0)
            })

            # Save variations
            for i, variation in enumerate(variations):
                variation_filename = f"beat_{workflow_id}_var_{i + 1}_{variation['strategy']}.wav"
                variation_path = Path(settings.BEATS_PATH) / variation_filename

                # TODO: Implement actual audio saving
                variation_path.touch()

                beat_files.append({
                    "type": "variation",
                    "strategy": variation["strategy"],
                    "audio_path": str(variation_path),
                    "metadata": variation["metadata"],
                    "quality_score": variation.get("quality_score", 7.0),
                    "variation_parameters": variation.get("variation_parameters", {})
                })

            return Result.ok(beat_files)

        except Exception as e:
            return Result.err(f"Failed to save beats: {e}")

    async def _export_midi_for_beats(
        self,
        beat_files: List[Dict[str, Any]],
        tempo: float,
        time_signature: str
    ) -> Optional[Dict[str, str]]:
        """Export MIDI files for all beats"""

        midi_results = {}

        try:
            for beat_file in beat_files:
                audio_path = beat_file["audio_path"]
                midi_filename = Path(audio_path).stem + ".mid"
                midi_path = Path(audio_path).parent / midi_filename

                midi_result = await self.midi_service.convert_beat_to_midi(
                    audio_path,
                    str(midi_path),
                    tempo,
                    time_signature
                )

                if midi_result.success:
                    midi_results[audio_path] = str(midi_path)
                else:
                    audio_logger.log_processing_error(
                        operation="MIDIExportError",
                        error=f"MIDI export failed for {audio_path}: {midi_result.error}"
                    )

            return midi_results

        except Exception as e:
            audio_logger.log_processing_error(
                operation="MIDIExportError",
                error=f"MIDI export process failed: {e}"
            )
            return None

    async def _integrate_beats_into_project(
        self,
        project_id: uuid.UUID,
        beat_files: List[Dict[str, Any]],
        midi_results: Optional[Dict[str, str]],
        track_id: Optional[uuid.UUID],
        request: BeatGenerationRequest
    ) -> Result[Dict[str, Any]]:
        """Integrate generated beats into project"""

        try:
            integration_results = {
                "track_ids": [],
                "clip_ids": [],
                "primary_clip_id": None,
                "variation_clip_ids": []
            }

            # Create or use existing track
            if not track_id:
                track_result = await self.integration_service.create_beat_track_template(
                    project_id,
                    f"Generated Beats - {request.prompt[:20]}..."
                )
                if not track_result.success:
                    return Result.err(f"Failed to create beat track: {track_result.error}")
                track_id = track_result.data

            integration_results["track_ids"].append(track_id)

            # Add beats to track
            timeline_position = 0.0
            for i, beat_file in enumerate(beat_files):
                clip_name = f"{beat_file.get('type', 'beat')} {i + 1}"
                if beat_file.get("strategy"):
                    clip_name += f" ({beat_file['strategy']})"

                clip_result = await self.integration_service.add_beat_with_midi_export(
                    beat_file["audio_path"],
                    track_id,
                    timeline_position,
                    float(request.tempo),
                    request.time_signature,
                    clip_name,
                    auto_export_midi=False  # We already have MIDI files
                )

                if clip_result.success:
                    clip_id = clip_result.data["clip_id"]
                    integration_results["clip_ids"].append(clip_id)

                    if beat_file.get("type") == "primary":
                        integration_results["primary_clip_id"] = clip_id
                    else:
                        integration_results["variation_clip_ids"].append(clip_id)

                # Space clips apart (could be customizable)
                timeline_position += float(request.duration) + 2.0  # 2 second gap

            return Result.ok(integration_results)

        except Exception as e:
            return Result.err(f"Project integration failed: {e}")

    async def create_beat_library_from_template(
        self,
        project_id: uuid.UUID,
        template_style: str,
        tempo_range: tuple = (100, 140),
        variation_count: int = 5,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Result[Dict[str, Any]]:
        """Create a library of beats based on a template style"""

        try:
            audio_logger.log_processing_start(
                operation="CreateBeatLibrary",
                project_id=str(project_id),
                template_style=template_style,
                variation_count=variation_count
            )

            if progress_callback:
                progress_callback(0.1, f"Creating {template_style} beat library")

            # Generate tempo variations
            min_tempo, max_tempo = tempo_range
            tempo_step = (max_tempo - min_tempo) / max(variation_count - 1, 1)

            library_beats = []
            track_organization = {}

            for i in range(variation_count):
                tempo = min_tempo + (i * tempo_step)
                tempo = round(tempo, 0)

                if progress_callback:
                    progress_callback(
                        0.2 + (0.6 * i / variation_count),
                        f"Generating {template_style} beat at {tempo} BPM"
                    )

                # Create request for this variation
                request = BeatGenerationRequest(
                    project_id=project_id,
                    prompt=f"{template_style} drum beat",
                    provider="musicgen",
                    duration=Decimal("8.0"),
                    tempo=Decimal(str(tempo)),
                    time_signature="4/4",
                    style_tags={"genre": template_style, "tempo_category": self._get_tempo_category(tempo)}
                )

                # Generate beat
                workflow_result = await self.execute_complete_beat_workflow(
                    request,
                    project_id,
                    uuid.uuid4(),  # Placeholder user ID
                    auto_add_to_project=False,
                    generate_variations=1
                )

                if workflow_result.success:
                    library_beats.append({
                        "tempo": tempo,
                        "workflow_data": workflow_result.data
                    })

            # Organize into project tracks by tempo category
            if progress_callback:
                progress_callback(0.9, "Organizing beats into project tracks")

            track_organization = await self._organize_library_by_tempo(
                project_id, library_beats, template_style
            )

            if progress_callback:
                progress_callback(1.0, f"Beat library created with {len(library_beats)} beats")

            audio_logger.log_processing_complete(
                operation="CreateBeatLibrary",
                duration_ms=len(library_beats) * 10000,  # Estimate
                beats_generated=len(library_beats),
                tracks_created=len(track_organization)
            )

            return Result.ok({
                "library_style": template_style,
                "beats_generated": len(library_beats),
                "tempo_range": tempo_range,
                "track_organization": track_organization,
                "beat_details": library_beats
            })

        except Exception as e:
            audio_logger.log_processing_error(
                operation="CreateBeatLibrary",
                error=str(e)
            )
            return Result.err(f"Beat library creation failed: {e}")

    def _get_tempo_category(self, tempo: float) -> str:
        """Categorize tempo for organization"""
        if tempo < 80:
            return "slow"
        elif tempo < 100:
            return "medium_slow"
        elif tempo < 120:
            return "medium"
        elif tempo < 140:
            return "medium_fast"
        else:
            return "fast"

    async def _organize_library_by_tempo(
        self,
        project_id: uuid.UUID,
        library_beats: List[Dict[str, Any]],
        style: str
    ) -> Dict[str, Any]:
        """Organize library beats into tracks by tempo category"""

        tempo_tracks = {}

        # Group by tempo category
        tempo_groups = {}
        for beat_data in library_beats:
            tempo = beat_data["tempo"]
            category = self._get_tempo_category(tempo)

            if category not in tempo_groups:
                tempo_groups[category] = []
            tempo_groups[category].append(beat_data)

        # Create tracks and add beats
        for category, beats in tempo_groups.items():
            track_name = f"{style.title()} Beats - {category.replace('_', ' ').title()}"

            track_result = await self.integration_service.create_beat_track_template(
                project_id, track_name
            )

            if track_result.success:
                track_id = track_result.data
                tempo_tracks[category] = {
                    "track_id": track_id,
                    "track_name": track_name,
                    "beat_count": len(beats),
                    "tempo_range": (
                        min(b["tempo"] for b in beats),
                        max(b["tempo"] for b in beats)
                    )
                }

        return tempo_tracks

    async def cleanup_workflow_files(self, workflow_id: uuid.UUID) -> Result[bool]:
        """Clean up temporary files from workflow"""

        try:
            from ..core.config import get_settings
            settings = get_settings()

            beats_dir = Path(settings.BEATS_PATH)
            midi_dir = Path(settings.MIDI_PATH)

            # Find and remove workflow files
            workflow_pattern = f"*{workflow_id}*"

            for directory in [beats_dir, midi_dir]:
                for file_path in directory.glob(workflow_pattern):
                    if file_path.is_file():
                        file_path.unlink()

            return Result.ok(True)

        except Exception as e:
            return Result.err(f"Cleanup failed: {e}")