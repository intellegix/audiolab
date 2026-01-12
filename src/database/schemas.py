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


# Update forward references for recursive models
ProjectDetail.model_rebuild()
TrackDetail.model_rebuild()
ClipDetail.model_rebuild()