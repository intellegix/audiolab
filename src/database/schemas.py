"""
AudioLab Pydantic Schemas
Request/response models for API validation and serialization
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# Base configuration for all schemas
class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
        validate_assignment=True,
        arbitrary_types_allowed=True
    )


# Project Schemas
class ProjectBase(BaseSchema):
    """Base project fields"""
    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    sample_rate: int = Field(default=48000, ge=8000, le=192000, description="Audio sample rate in Hz")
    bit_depth: int = Field(default=24, ge=16, le=32, description="Audio bit depth")
    tempo: Decimal = Field(default=Decimal("120.0"), ge=Decimal("20.0"), le=Decimal("999.99"), description="Tempo in BPM")
    time_signature: str = Field(default="4/4", pattern=r"^\d+/\d+$", description="Time signature")


class ProjectCreate(ProjectBase):
    """Schema for creating a project"""
    user_id: uuid.UUID = Field(..., description="ID of the user creating the project")


class ProjectUpdate(BaseSchema):
    """Schema for updating a project"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    sample_rate: Optional[int] = Field(None, ge=8000, le=192000)
    bit_depth: Optional[int] = Field(None, ge=16, le=32)
    tempo: Optional[Decimal] = Field(None, ge=Decimal("20.0"), le=Decimal("999.99"))
    time_signature: Optional[str] = Field(None, pattern=r"^\d+/\d+$")


class ProjectResponse(ProjectBase):
    """Schema for project responses"""
    id: uuid.UUID
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class ProjectDetail(ProjectResponse):
    """Schema for detailed project view with relationships"""
    tracks: List["TrackResponse"] = Field(default_factory=list)


# Track Schemas
class TrackBase(BaseSchema):
    """Base track fields"""
    name: str = Field(..., min_length=1, max_length=255, description="Track name")
    track_index: int = Field(..., ge=0, le=31, description="Track position (0-31)")
    volume: Decimal = Field(default=Decimal("1.0"), ge=Decimal("0.0"), le=Decimal("2.0"), description="Track volume multiplier")
    pan: Decimal = Field(default=Decimal("0.0"), ge=Decimal("-1.0"), le=Decimal("1.0"), description="Pan position (-1.0 left to 1.0 right)")
    muted: bool = Field(default=False, description="Track mute state")
    soloed: bool = Field(default=False, description="Track solo state")
    color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$", description="Track color (hex format)")


class TrackCreate(TrackBase):
    """Schema for creating a track"""
    project_id: uuid.UUID = Field(..., description="ID of the parent project")


class TrackUpdate(BaseSchema):
    """Schema for updating a track"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    track_index: Optional[int] = Field(None, ge=0, le=31)
    volume: Optional[Decimal] = Field(None, ge=Decimal("0.0"), le=Decimal("2.0"))
    pan: Optional[Decimal] = Field(None, ge=Decimal("-1.0"), le=Decimal("1.0"))
    muted: Optional[bool] = None
    soloed: Optional[bool] = None
    color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")


class TrackResponse(TrackBase):
    """Schema for track responses"""
    id: uuid.UUID
    project_id: uuid.UUID
    created_at: datetime


class TrackDetail(TrackResponse):
    """Schema for detailed track view with relationships"""
    clips: List["ClipResponse"] = Field(default_factory=list)
    effects: List["EffectResponse"] = Field(default_factory=list)


# Clip Schemas
class ClipBase(BaseSchema):
    """Base clip fields"""
    name: Optional[str] = Field(None, max_length=255, description="Clip name")
    file_path: str = Field(..., description="Path to audio file")
    start_time: Decimal = Field(..., ge=Decimal("0.0"), description="Start time on timeline in seconds")
    duration: Decimal = Field(..., gt=Decimal("0.0"), description="Clip duration in seconds")
    offset: Decimal = Field(default=Decimal("0.0"), ge=Decimal("0.0"), description="Offset into source file in seconds")
    fade_in: Decimal = Field(default=Decimal("0.0"), ge=Decimal("0.0"), le=Decimal("10.0"), description="Fade in duration in seconds")
    fade_out: Decimal = Field(default=Decimal("0.0"), ge=Decimal("0.0"), le=Decimal("10.0"), description="Fade out duration in seconds")
    gain: Decimal = Field(default=Decimal("0.0"), ge=Decimal("-60.0"), le=Decimal("20.0"), description="Gain in dB")


class ClipCreate(ClipBase):
    """Schema for creating a clip"""
    track_id: uuid.UUID = Field(..., description="ID of the parent track")


class ClipUpdate(BaseSchema):
    """Schema for updating a clip"""
    name: Optional[str] = Field(None, max_length=255)
    start_time: Optional[Decimal] = Field(None, ge=Decimal("0.0"))
    duration: Optional[Decimal] = Field(None, gt=Decimal("0.0"))
    offset: Optional[Decimal] = Field(None, ge=Decimal("0.0"))
    fade_in: Optional[Decimal] = Field(None, ge=Decimal("0.0"), le=Decimal("10.0"))
    fade_out: Optional[Decimal] = Field(None, ge=Decimal("0.0"), le=Decimal("10.0"))
    gain: Optional[Decimal] = Field(None, ge=Decimal("-60.0"), le=Decimal("20.0"))


class ClipResponse(ClipBase):
    """Schema for clip responses"""
    id: uuid.UUID
    track_id: uuid.UUID
    created_at: datetime


class ClipDetail(ClipResponse):
    """Schema for detailed clip view with relationships"""
    effects: List["EffectResponse"] = Field(default_factory=list)
    stem_separations: List["StemSeparationResponse"] = Field(default_factory=list)


# Effect Schemas
class EffectBase(BaseSchema):
    """Base effect fields"""
    name: str = Field(..., min_length=1, max_length=100, description="Effect name")
    effect_type: str = Field(..., min_length=1, max_length=50, description="Effect type")
    parameters: Dict[str, Any] = Field(..., description="Effect parameters (flexible JSON)")
    bypass: bool = Field(default=False, description="Effect bypass state")
    order_index: int = Field(..., ge=0, le=7, description="Position in effect chain (0-7)")


class EffectCreate(EffectBase):
    """Schema for creating an effect"""
    track_id: Optional[uuid.UUID] = Field(None, description="ID of the parent track")
    clip_id: Optional[uuid.UUID] = Field(None, description="ID of the parent clip")

    def validate_parent(self):
        """Validate that either track_id or clip_id is set, but not both"""
        if not self.track_id and not self.clip_id:
            raise ValueError("Either track_id or clip_id must be provided")
        if self.track_id and self.clip_id:
            raise ValueError("Cannot specify both track_id and clip_id")
        return self


class EffectUpdate(BaseSchema):
    """Schema for updating an effect"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    effect_type: Optional[str] = Field(None, min_length=1, max_length=50)
    parameters: Optional[Dict[str, Any]] = None
    bypass: Optional[bool] = None
    order_index: Optional[int] = Field(None, ge=0, le=7)


class EffectResponse(EffectBase):
    """Schema for effect responses"""
    id: uuid.UUID
    track_id: Optional[uuid.UUID]
    clip_id: Optional[uuid.UUID]
    created_at: datetime


# Stem Separation Schemas
class StemSeparationBase(BaseSchema):
    """Base stem separation fields"""
    stems: Dict[str, str] = Field(..., description="Stem file paths by type")
    model_used: str = Field(default="htdemucs_ft", description="Demucs model used")
    processing_time: Optional[Decimal] = Field(None, ge=Decimal("0.0"), description="Processing time in seconds")
    quality_score: Optional[Decimal] = Field(None, ge=Decimal("0.0"), le=Decimal("20.0"), description="SDR quality score")


class StemSeparationCreate(StemSeparationBase):
    """Schema for creating a stem separation"""
    clip_id: uuid.UUID = Field(..., description="ID of the source clip")


class StemSeparationUpdate(BaseSchema):
    """Schema for updating a stem separation"""
    stems: Optional[Dict[str, str]] = None
    processing_time: Optional[Decimal] = Field(None, ge=Decimal("0.0"))
    quality_score: Optional[Decimal] = Field(None, ge=Decimal("0.0"), le=Decimal("20.0"))


class StemSeparationResponse(StemSeparationBase):
    """Schema for stem separation responses"""
    id: uuid.UUID
    clip_id: uuid.UUID
    created_at: datetime


# Audio Processing Schemas
class AudioProcessingRequest(BaseSchema):
    """Schema for audio processing requests"""
    operation: str = Field(..., description="Processing operation type")
    input_file_path: str = Field(..., description="Path to input audio file")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")


class AudioProcessingResponse(BaseSchema):
    """Schema for audio processing responses"""
    success: bool = Field(..., description="Processing success status")
    message: str = Field(..., description="Processing result message")
    result_path: Optional[str] = Field(None, description="Path to processed audio file")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")


# Export Schemas
class ExportRequest(BaseSchema):
    """Schema for audio export requests"""
    project_id: uuid.UUID = Field(..., description="Project to export")
    format: str = Field(default="wav", description="Export format")
    quality: str = Field(default="24bit", description="Export quality")
    lufs_target: Optional[Decimal] = Field(None, description="Target LUFS loudness")


class ExportResponse(BaseSchema):
    """Schema for export responses"""
    success: bool = Field(..., description="Export success status")
    file_path: Optional[str] = Field(None, description="Path to exported file")
    lufs_measured: Optional[Decimal] = Field(None, description="Measured LUFS loudness")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Export metadata")


# Error Schemas
class ErrorDetail(BaseSchema):
    """Schema for error details"""
    field: Optional[str] = Field(None, description="Field that caused the error")
    message: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseSchema):
    """Schema for error responses"""
    success: bool = Field(default=False, description="Operation success status")
    error: str = Field(..., description="Error message")
    details: Optional[List[ErrorDetail]] = Field(None, description="Detailed error information")


# Advanced Separation Schemas
class AdvancedSeparationBase(BaseSchema):
    """Base advanced separation fields"""
    separation_type: str = Field(default="advanced", description="Type of separation performed")
    target_instruments: List[str] = Field(..., description="List of instruments to separate individually")
    instruments_separated: Dict[str, Dict[str, Any]] = Field(..., description="Separated instrument files and metadata")
    enhancement_applied: Optional[Dict[str, List[str]]] = Field(None, description="Enhancement applied to each instrument")
    quality_metrics: Dict[str, float] = Field(..., description="Quality metrics for separation")
    model_used: str = Field(default="htdemucs_6s", description="AI model used for separation")
    processing_time: Optional[Decimal] = Field(None, ge=Decimal("0.0"), description="Processing time in seconds")
    algorithm_version: str = Field(default="1.0", description="Algorithm version used")
    processing_parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters used for processing")


class AdvancedSeparationCreate(AdvancedSeparationBase):
    """Schema for creating an advanced separation"""
    clip_id: uuid.UUID = Field(..., description="ID of the source clip")


class AdvancedSeparationUpdate(BaseSchema):
    """Schema for updating an advanced separation"""
    enhancement_applied: Optional[Dict[str, List[str]]] = None
    quality_metrics: Optional[Dict[str, float]] = None
    processing_time: Optional[Decimal] = Field(None, ge=Decimal("0.0"))


class AdvancedSeparationResponse(AdvancedSeparationBase):
    """Schema for advanced separation responses"""
    id: uuid.UUID
    clip_id: uuid.UUID
    created_at: datetime


# Advanced Separation Request Schemas
class AdvancedSeparationRequest(BaseSchema):
    """Schema for requesting advanced separation"""
    instruments: List[str] = Field(..., description="Instruments to separate (vocals, rhythm_guitar, lead_guitar, etc.)")
    model: str = Field(default="htdemucs_6s", description="Model to use for separation")
    enhancement_options: Optional[Dict[str, Any]] = Field(None, description="Optional enhancement parameters")


class AdvancedSeparationJobResponse(BaseSchema):
    """Schema for advanced separation job creation response"""
    job_id: uuid.UUID = Field(..., description="Processing job ID")
    estimated_duration: Optional[int] = Field(None, description="Estimated processing time in seconds")
    status: str = Field(..., description="Initial job status")


# Enhancement Profile Schemas
class EnhancementProfileBase(BaseSchema):
    """Base enhancement profile fields"""
    profile_name: str = Field(..., min_length=1, max_length=255, description="Profile name")
    description: Optional[str] = Field(None, description="Profile description")
    is_default: bool = Field(default=False, description="Whether this is the user's default profile")
    is_public: bool = Field(default=False, description="Whether this profile is publicly available")
    enhancement_settings: Dict[str, Dict[str, Any]] = Field(..., description="Enhancement settings per instrument type")
    global_settings: Optional[Dict[str, Any]] = Field(None, description="Global enhancement parameters")


class EnhancementProfileCreate(EnhancementProfileBase):
    """Schema for creating an enhancement profile"""
    user_id: uuid.UUID = Field(..., description="ID of the user creating the profile")


class EnhancementProfileUpdate(BaseSchema):
    """Schema for updating an enhancement profile"""
    profile_name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    is_default: Optional[bool] = None
    is_public: Optional[bool] = None
    enhancement_settings: Optional[Dict[str, Dict[str, Any]]] = None
    global_settings: Optional[Dict[str, Any]] = None


class EnhancementProfileResponse(EnhancementProfileBase):
    """Schema for enhancement profile responses"""
    id: uuid.UUID
    user_id: uuid.UUID
    usage_count: int
    last_used_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime


# Enhancement Request Schemas
class EnhancementRequest(BaseSchema):
    """Schema for requesting audio enhancement"""
    stems: List[str] = Field(..., description="Stems to enhance (vocals, guitar, etc.)")
    enhancements: List[str] = Field(..., description="Enhancement types to apply (denoise, clarity, upsampl, etc.)")
    level: float = Field(default=0.7, ge=0.0, le=1.0, description="Enhancement intensity level")
    profile_id: Optional[uuid.UUID] = Field(None, description="Enhancement profile to use")


class EnhancementResponse(BaseSchema):
    """Schema for enhancement results"""
    enhanced_stems: Dict[str, Dict[str, Any]] = Field(..., description="Enhanced stem files and metadata")
    quality_improvements: Dict[str, float] = Field(..., description="Quality improvement scores per stem")
    processing_time: Decimal = Field(..., description="Total enhancement processing time")


class EnhancementPreviewRequest(BaseSchema):
    """Schema for requesting enhancement preview"""
    stem: str = Field(..., description="Stem to preview enhancement for")
    enhancement: str = Field(..., description="Enhancement type to preview")
    level: float = Field(default=0.5, ge=0.0, le=1.0, description="Enhancement level")
    preview_duration: int = Field(default=30, ge=5, le=60, description="Preview duration in seconds")


class EnhancementPreviewResponse(BaseSchema):
    """Schema for enhancement preview results"""
    preview_url: str = Field(..., description="URL to preview audio file")
    duration: int = Field(..., description="Preview duration in seconds")
    quality_preview: Dict[str, float] = Field(..., description="Preview quality metrics")


# Processing Job Schemas
class ProcessingJobBase(BaseSchema):
    """Base processing job fields"""
    job_type: str = Field(..., description="Type of processing job")
    status: str = Field(default="pending", description="Current job status")
    progress: Decimal = Field(default=Decimal("0.0"), ge=Decimal("0.0"), le=Decimal("100.0"), description="Progress percentage")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    estimated_duration: Optional[int] = Field(None, ge=0, description="Estimated duration in seconds")
    input_parameters: Optional[Dict[str, Any]] = Field(None, description="Job input parameters")
    result_data: Optional[Dict[str, Any]] = Field(None, description="Job result data")
    error_message: Optional[str] = Field(None, description="Error message if job failed")


class ProcessingJobCreate(ProcessingJobBase):
    """Schema for creating a processing job"""
    user_id: uuid.UUID = Field(..., description="ID of the user creating the job")
    clip_id: Optional[uuid.UUID] = Field(None, description="ID of the clip being processed")


class ProcessingJobUpdate(BaseSchema):
    """Schema for updating a processing job"""
    status: Optional[str] = None
    progress: Optional[Decimal] = Field(None, ge=Decimal("0.0"), le=Decimal("100.0"))
    current_stage: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class ProcessingJobResponse(ProcessingJobBase):
    """Schema for processing job responses"""
    id: uuid.UUID
    user_id: uuid.UUID
    clip_id: Optional[uuid.UUID]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


# Quality Analysis Schemas
class QualityAnalysisRequest(BaseSchema):
    """Schema for requesting quality analysis"""
    include_stems: bool = Field(default=True, description="Include per-stem analysis")
    detailed_metrics: bool = Field(default=False, description="Include detailed quality metrics")


class QualityAnalysisResponse(BaseSchema):
    """Schema for quality analysis results"""
    overall_score: float = Field(..., ge=0.0, le=10.0, description="Overall quality score")
    stems: Dict[str, Dict[str, float]] = Field(..., description="Per-stem quality metrics")
    analysis_metadata: Dict[str, Any] = Field(..., description="Analysis metadata and timestamps")


# Beat Generation Schemas
class BeatGenerationBase(BaseSchema):
    """Base beat generation fields"""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for beat generation")
    provider: str = Field(default="musicgen", description="AI provider for generation")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    duration: Decimal = Field(..., gt=Decimal("0.0"), le=Decimal("300.0"), description="Generated beat duration in seconds")
    tempo: Decimal = Field(..., ge=Decimal("60.0"), le=Decimal("200.0"), description="Target tempo in BPM")
    time_signature: str = Field(..., pattern=r"^\d+/\d+$", description="Time signature (e.g., '4/4')")
    style_tags: Optional[Dict[str, Any]] = Field(None, description="Style and genre tags")


class BeatGenerationRequest(BeatGenerationBase):
    """Schema for creating a beat generation request"""
    project_id: uuid.UUID = Field(..., description="ID of the target project")


class BeatGenerationUpdate(BaseSchema):
    """Schema for updating beat generation status"""
    status: Optional[str] = None
    progress: Optional[Decimal] = Field(None, ge=Decimal("0.0"), le=Decimal("100.0"))
    current_stage: Optional[str] = None
    generated_audio_path: Optional[str] = None
    generated_midi_path: Optional[str] = None
    quality_score: Optional[Decimal] = Field(None, ge=Decimal("0.0"), le=Decimal("10.0"))
    processing_time: Optional[Decimal] = Field(None, ge=Decimal("0.0"))
    provider_metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class BeatGenerationResponse(BeatGenerationBase):
    """Schema for beat generation responses"""
    id: uuid.UUID
    project_id: uuid.UUID
    user_id: uuid.UUID
    status: str
    progress: Decimal
    current_stage: Optional[str]
    generated_audio_path: Optional[str]
    generated_midi_path: Optional[str]
    quality_score: Optional[Decimal]
    processing_time: Optional[Decimal]
    provider_metadata: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]


class BeatGenerationJobResponse(BaseSchema):
    """Schema for beat generation job creation response"""
    request_id: uuid.UUID = Field(..., description="Beat generation request ID")
    estimated_duration: Optional[int] = Field(None, description="Estimated processing time in seconds")
    status: str = Field(..., description="Initial job status")


# Beat Template Schemas
class BeatTemplateBase(BaseSchema):
    """Base beat template fields"""
    name: str = Field(..., min_length=1, max_length=255, description="Template name")
    description: Optional[str] = Field(None, description="Template description")
    category: str = Field(..., description="Beat category (hip-hop, rock, etc.)")
    tags: List[str] = Field(..., description="Searchable tags")
    default_tempo: Decimal = Field(..., ge=Decimal("60.0"), le=Decimal("200.0"), description="Default tempo in BPM")
    time_signature: str = Field(..., pattern=r"^\d+/\d+$", description="Time signature")
    duration: Decimal = Field(..., gt=Decimal("0.0"), le=Decimal("300.0"), description="Default duration")
    provider_config: Dict[str, Any] = Field(..., description="Provider-specific configuration")
    prompt_template: str = Field(..., description="Base prompt template")
    is_public: bool = Field(default=True, description="Whether template is publicly available")


class BeatTemplateCreate(BeatTemplateBase):
    """Schema for creating a beat template"""
    created_by_user_id: Optional[uuid.UUID] = Field(None, description="ID of user creating template")


class BeatTemplateUpdate(BaseSchema):
    """Schema for updating a beat template"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    default_tempo: Optional[Decimal] = Field(None, ge=Decimal("60.0"), le=Decimal("200.0"))
    time_signature: Optional[str] = Field(None, pattern=r"^\d+/\d+$")
    duration: Optional[Decimal] = Field(None, gt=Decimal("0.0"), le=Decimal("300.0"))
    provider_config: Optional[Dict[str, Any]] = None
    prompt_template: Optional[str] = None
    is_public: Optional[bool] = None


class BeatTemplateResponse(BeatTemplateBase):
    """Schema for beat template responses"""
    id: uuid.UUID
    usage_count: int
    average_quality: Optional[Decimal]
    created_by_user_id: Optional[uuid.UUID]
    created_at: datetime
    updated_at: datetime


# Beat Variation Schemas
class BeatVariationBase(BaseSchema):
    """Base beat variation fields"""
    variation_index: int = Field(..., ge=1, description="Variation number")
    name: Optional[str] = Field(None, max_length=255, description="Variation name")
    audio_path: str = Field(..., description="Path to generated audio file")
    midi_path: Optional[str] = Field(None, description="Path to generated MIDI file")
    quality_score: Optional[Decimal] = Field(None, ge=Decimal("0.0"), le=Decimal("10.0"), description="Quality score")
    user_rating: Optional[int] = Field(None, ge=1, le=5, description="User rating (1-5 stars)")
    generation_seed: Optional[int] = Field(None, description="Generation seed for reproducibility")
    generation_metadata: Optional[Dict[str, Any]] = Field(None, description="Generation metadata")
    is_selected: bool = Field(default=False, description="Whether this is the selected variation")
    used_in_project: bool = Field(default=False, description="Whether variation is used in project")


class BeatVariationCreate(BeatVariationBase):
    """Schema for creating a beat variation"""
    beat_generation_request_id: uuid.UUID = Field(..., description="ID of parent beat generation request")


class BeatVariationUpdate(BaseSchema):
    """Schema for updating a beat variation"""
    name: Optional[str] = Field(None, max_length=255)
    quality_score: Optional[Decimal] = Field(None, ge=Decimal("0.0"), le=Decimal("10.0"))
    user_rating: Optional[int] = Field(None, ge=1, le=5)
    is_selected: Optional[bool] = None
    used_in_project: Optional[bool] = None


class BeatVariationResponse(BeatVariationBase):
    """Schema for beat variation responses"""
    id: uuid.UUID
    beat_generation_request_id: uuid.UUID
    created_at: datetime


# Beat Generation with Variations Response
class BeatGenerationDetailResponse(BeatGenerationResponse):
    """Schema for detailed beat generation response with variations"""
    variations: List[BeatVariationResponse] = Field(default_factory=list, description="Generated variations")


# Beat Project Integration Schemas
class BeatToProjectRequest(BaseSchema):
    """Schema for adding generated beat to project"""
    track_id: uuid.UUID = Field(..., description="Target track ID")
    timeline_position: Decimal = Field(default=Decimal("0.0"), ge=Decimal("0.0"), description="Timeline position in seconds")
    variation_id: Optional[uuid.UUID] = Field(None, description="Specific variation ID to use")
    clip_name: Optional[str] = Field(None, max_length=255, description="Name for the created clip")


class BeatToProjectResponse(BaseSchema):
    """Schema for beat-to-project operation response"""
    success: bool = Field(..., description="Operation success status")
    clip_id: Optional[uuid.UUID] = Field(None, description="ID of created clip")
    message: str = Field(..., description="Operation result message")


# Beat Generation Progress Event (for WebSocket)
class BeatGenerationProgressEvent(BaseSchema):
    """Schema for beat generation progress WebSocket events"""
    request_id: uuid.UUID = Field(..., description="Beat generation request ID")
    status: str = Field(..., description="Current status")
    progress: Decimal = Field(..., description="Progress percentage")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    estimated_time_remaining: Optional[int] = Field(None, description="Estimated seconds remaining")
    error_message: Optional[str] = Field(None, description="Error message if failed")


# Update forward references for recursive models
ProjectDetail.model_rebuild()
TrackDetail.model_rebuild()
ClipDetail.model_rebuild()