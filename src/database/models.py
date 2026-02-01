"""
AudioLab Database Models
SQLAlchemy ORM models for all entities in the AudioLab application
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Optional, List
from sqlalchemy import (
    Column,
    String,
    Integer,
    Boolean,
    Text,
    TIMESTAMP,
    ForeignKey,
    Index,
    DECIMAL
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, Mapped, mapped_column

from .connection import Base


class Project(Base):
    """Audio project model - root entity for all audio work"""
    __tablename__ = "projects"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Project metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, default=48000, nullable=False)
    bit_depth: Mapped[int] = mapped_column(Integer, default=24, nullable=False)
    tempo: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), default=Decimal("120.0"), nullable=False)
    time_signature: Mapped[str] = mapped_column(String(10), default="4/4", nullable=False)

    # User and ownership
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    tracks: Mapped[List["Track"]] = relationship(
        "Track",
        back_populates="project",
        cascade="all, delete-orphan",
        order_by="Track.track_index"
    )
    loop_regions: Mapped[List["LoopRegion"]] = relationship(
        "LoopRegion",
        foreign_keys="LoopRegion.project_id",
        cascade="all, delete-orphan",
        order_by="LoopRegion.start_time"
    )

    def __repr__(self) -> str:
        return f"<Project(id={self.id}, name='{self.name}', sample_rate={self.sample_rate})>"


class Track(Base):
    """Audio track model - container for audio clips and effects"""
    __tablename__ = "tracks"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Foreign key to project
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False
    )

    # Track metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    track_index: Mapped[int] = mapped_column(Integer, nullable=False)

    # Audio parameters
    volume: Mapped[Decimal] = mapped_column(DECIMAL(3, 2), default=Decimal("1.0"), nullable=False)
    pan: Mapped[Decimal] = mapped_column(DECIMAL(3, 2), default=Decimal("0.0"), nullable=False)
    muted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    soloed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Visual
    color: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)  # Hex color code

    # Recording settings
    record_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    input_device_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    monitoring_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project", back_populates="tracks")
    clips: Mapped[List["Clip"]] = relationship(
        "Clip",
        back_populates="track",
        cascade="all, delete-orphan",
        order_by="Clip.start_time"
    )
    effects: Mapped[List["Effect"]] = relationship(
        "Effect",
        back_populates="track",
        cascade="all, delete-orphan",
        primaryjoin="Track.id == Effect.track_id",
        order_by="Effect.order_index"
    )
    recording_sessions: Mapped[List["RecordingSession"]] = relationship(
        "RecordingSession",
        foreign_keys="RecordingSession.track_id",
        cascade="all, delete-orphan",
        order_by="RecordingSession.created_at"
    )

    def __repr__(self) -> str:
        return f"<Track(id={self.id}, name='{self.name}', project_id={self.project_id})>"


class Clip(Base):
    """Audio clip model - represents audio regions on timeline"""
    __tablename__ = "clips"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Foreign key to track
    track_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tracks.id", ondelete="CASCADE"),
        nullable=False
    )

    # Clip metadata
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)

    # Timeline positioning (in seconds with microsecond precision)
    start_time: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)
    duration: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)
    offset: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), default=Decimal("0.0"), nullable=False)

    # Audio processing
    fade_in: Mapped[Decimal] = mapped_column(DECIMAL(6, 3), default=Decimal("0.0"), nullable=False)
    fade_out: Mapped[Decimal] = mapped_column(DECIMAL(6, 3), default=Decimal("0.0"), nullable=False)
    gain: Mapped[Decimal] = mapped_column(DECIMAL(6, 2), default=Decimal("0.0"), nullable=False)  # dB

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )

    # Relationships
    track: Mapped["Track"] = relationship("Track", back_populates="clips")
    effects: Mapped[List["Effect"]] = relationship(
        "Effect",
        back_populates="clip",
        cascade="all, delete-orphan",
        primaryjoin="Clip.id == Effect.clip_id",
        order_by="Effect.order_index"
    )
    stem_separations: Mapped[List["StemSeparation"]] = relationship(
        "StemSeparation",
        back_populates="clip",
        cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Clip(id={self.id}, name='{self.name}', track_id={self.track_id})>"


class Effect(Base):
    """Audio effect model - represents processing applied to tracks or clips"""
    __tablename__ = "effects"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Foreign keys (either track_id OR clip_id, not both)
    track_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tracks.id", ondelete="CASCADE"),
        nullable=True
    )
    clip_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clips.id", ondelete="CASCADE"),
        nullable=True
    )

    # Effect metadata
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    effect_type: Mapped[str] = mapped_column(String(50), nullable=False)  # eq, compressor, reverb, etc.

    # Effect configuration (flexible JSON storage)
    parameters: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Effect state
    bypass: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)  # Position in effect chain

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )

    # Relationships
    track: Mapped[Optional["Track"]] = relationship(
        "Track",
        back_populates="effects",
        foreign_keys=[track_id]
    )
    clip: Mapped[Optional["Clip"]] = relationship(
        "Clip",
        back_populates="effects",
        foreign_keys=[clip_id]
    )

    def __repr__(self) -> str:
        target = f"track_id={self.track_id}" if self.track_id else f"clip_id={self.clip_id}"
        return f"<Effect(id={self.id}, type='{self.effect_type}', {target})>"

    # Check constraint: either track_id or clip_id must be set
    __table_args__ = (
        Index("ix_effect_track_id", "track_id"),
        Index("ix_effect_clip_id", "clip_id"),
        Index("ix_effect_order", "track_id", "clip_id", "order_index"),
    )


class StemSeparation(Base):
    """AI stem separation results - stores Demucs processing outputs"""
    __tablename__ = "stem_separations"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Foreign key to clip
    clip_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clips.id", ondelete="CASCADE"),
        nullable=False
    )

    # Separation results (JSON with stem file paths)
    stems: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # AI processing metadata
    model_used: Mapped[str] = mapped_column(String(50), default="htdemucs_ft", nullable=False)
    processing_time: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(8, 3), nullable=True)  # seconds
    quality_score: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(4, 2), nullable=True)  # SDR score

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )

    # Relationships
    clip: Mapped["Clip"] = relationship("Clip", back_populates="stem_separations")

    def __repr__(self) -> str:
        return f"<StemSeparation(id={self.id}, model='{self.model_used}', clip_id={self.clip_id})>"


class RecordingSession(Base):
    """Recording session model - manages real-time recording state"""
    __tablename__ = "recording_sessions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Foreign key to track
    track_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tracks.id", ondelete="CASCADE"),
        nullable=False
    )

    # Recording parameters
    input_device_id: Mapped[str] = mapped_column(String(255), nullable=False)
    start_time: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)  # Timeline position in seconds
    duration: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(10, 6), nullable=True)  # Duration in seconds

    # Session status: recording, stopped, saved, error
    status: Mapped[str] = mapped_column(String(20), default="recording", nullable=False)

    # Audio parameters captured during recording
    sample_rate: Mapped[int] = mapped_column(Integer, default=48000, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, default=1, nullable=False)  # 1=mono, 2=stereo
    bit_depth: Mapped[int] = mapped_column(Integer, default=24, nullable=False)

    # File path where recorded audio is being saved
    temp_file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    final_clip_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clips.id", ondelete="SET NULL"),
        nullable=True
    )

    # Metadata
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    track: Mapped["Track"] = relationship("Track", foreign_keys=[track_id])
    final_clip: Mapped[Optional["Clip"]] = relationship("Clip", foreign_keys=[final_clip_id])

    def __repr__(self) -> str:
        return f"<RecordingSession(id={self.id}, track_id={self.track_id}, status='{self.status}')>"


class LoopRegion(Base):
    """Loop region model - defines timeline regions for looping playback"""
    __tablename__ = "loop_regions"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Foreign key to project
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False
    )

    # Loop region parameters
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="Loop Region")
    start_time: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)  # Start position in seconds
    end_time: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)    # End position in seconds
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # Loop behavior
    repeat_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # None = infinite loops
    auto_punch_record: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # Auto record in loop

    # Visual and organization
    color: Mapped[Optional[str]] = mapped_column(String(7), nullable=True)  # Hex color code
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

    # Relationships
    project: Mapped["Project"] = relationship("Project", foreign_keys=[project_id])

    def __repr__(self) -> str:
        return f"<LoopRegion(id={self.id}, name='{self.name}', start={self.start_time}, end={self.end_time})>"


class AdvancedSeparation(Base):
    """Advanced separation model - extends stem separation with individual instrument tracking"""
    __tablename__ = "advanced_separations"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Foreign key to clip
    clip_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clips.id", ondelete="CASCADE"),
        nullable=False
    )

    # Separation configuration
    separation_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="advanced"
    )  # "advanced", "enhanced", "custom"

    target_instruments: Mapped[List[str]] = mapped_column(
        JSONB,
        nullable=False
    )  # ["vocals", "rhythm_guitar", "lead_guitar", etc.]

    # Results - individual instrument files and metadata
    instruments_separated: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False
    )  # {"vocals": {"file_path": "...", "quality": 8.5}, "rhythm_guitar": {...}}

    # Enhancement applied (if any)
    enhancement_applied: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True
    )  # {"vocals": ["denoise", "clarity"], "rhythm_guitar": ["eq", "compression"]}

    # Quality metrics for this separation
    quality_metrics: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False
    )  # {"overall_score": 8.5, "vocals_quality": 9.2, "energy_preservation": 0.95}

    # Processing metadata
    model_used: Mapped[str] = mapped_column(String(50), default="htdemucs_6s", nullable=False)
    processing_time: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(8, 3), nullable=True)  # seconds
    algorithm_version: Mapped[str] = mapped_column(String(20), default="1.0", nullable=False)

    # Processing parameters
    processing_parameters: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True
    )  # Stores parameters used for separation

    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )

    # Relationships
    clip: Mapped["Clip"] = relationship("Clip")

    def __repr__(self) -> str:
        return f"<AdvancedSeparation(id={self.id}, type='{self.separation_type}', clip_id={self.clip_id})>"


class EnhancementProfile(Base):
    """Enhancement profile model - user preferences and presets for audio enhancement"""
    __tablename__ = "enhancement_profiles"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # User association
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Profile metadata
    profile_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Enhancement settings per instrument type
    enhancement_settings: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False
    )  # {"vocals": {"denoise": 0.7, "clarity": 0.8}, "guitar": {"eq": {...}}}

    # Global enhancement parameters
    global_settings: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True
    )  # {"master_level": 0.8, "quality_priority": "speed", "preview_length": 30}

    # Usage statistics
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<EnhancementProfile(id={self.id}, name='{self.profile_name}', user_id={self.user_id})>"


class ProcessingJob(Base):
    """Processing job model - tracks long-running audio processing tasks"""
    __tablename__ = "processing_jobs"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Job metadata
    job_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False
    )  # "advanced_separation", "enhancement", "stem_separation", "beat_generation"

    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False
    )  # "pending", "processing", "completed", "failed", "cancelled"

    # Progress tracking
    progress: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), default=Decimal("0.0"), nullable=False)  # 0.0 to 100.0
    current_stage: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    estimated_duration: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # seconds

    # Foreign keys to relevant entities
    clip_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("clips.id", ondelete="CASCADE"),
        nullable=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Job parameters
    input_parameters: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Results
    result_data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)

    # Relationships
    clip: Mapped[Optional["Clip"]] = relationship("Clip")

    def __repr__(self) -> str:
        return f"<ProcessingJob(id={self.id}, type='{self.job_type}', status='{self.status}')>"


class BeatGenerationRequest(Base):
    """Beat generation request model - tracks AI beat generation tasks"""
    __tablename__ = "beat_generation_requests"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Foreign keys
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)

    # Generation parameters
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    provider: Mapped[str] = mapped_column(String(20), default="musicgen", nullable=False)  # "musicgen", "soundraw"
    model_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    duration: Mapped[Decimal] = mapped_column(DECIMAL(6, 2), nullable=False)  # seconds

    # Musical synchronization (from project)
    tempo: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), nullable=False)  # BPM
    time_signature: Mapped[str] = mapped_column(String(10), nullable=False)  # e.g., "4/4"
    style_tags: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)  # genre, mood, etc.

    # Processing state
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)  # pending, processing, completed, failed
    progress: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), default=Decimal("0.0"), nullable=False)  # 0.0 to 100.0
    current_stage: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Results
    generated_audio_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    generated_midi_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    quality_score: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(4, 2), nullable=True)  # 0.0 to 10.0
    processing_time: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(8, 3), nullable=True)  # seconds
    provider_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP, nullable=True)

    # Relationships
    project: Mapped["Project"] = relationship("Project", foreign_keys=[project_id])

    def __repr__(self) -> str:
        return f"<BeatGenerationRequest(id={self.id}, provider='{self.provider}', status='{self.status}')>"


class BeatTemplate(Base):
    """Beat template model - predefined beat patterns and configurations"""
    __tablename__ = "beat_templates"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Template metadata
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(50), nullable=False)  # "hip-hop", "rock", "electronic", etc.
    tags: Mapped[List[str]] = mapped_column(JSONB, nullable=False)  # searchable tags

    # Musical parameters
    default_tempo: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), nullable=False)  # BPM
    time_signature: Mapped[str] = mapped_column(String(10), nullable=False)
    duration: Mapped[Decimal] = mapped_column(DECIMAL(6, 2), nullable=False)  # seconds

    # Generation parameters
    provider_config: Mapped[dict] = mapped_column(JSONB, nullable=False)  # provider-specific settings
    prompt_template: Mapped[str] = mapped_column(Text, nullable=False)  # base prompt with variables

    # Usage and quality metrics
    usage_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    average_quality: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(4, 2), nullable=True)
    is_public: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_by_user_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<BeatTemplate(id={self.id}, name='{self.name}', category='{self.category}')>"


class BeatVariation(Base):
    """Beat variation model - stores multiple generated versions of the same beat request"""
    __tablename__ = "beat_variations"

    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid()
    )

    # Foreign key to beat generation request
    beat_generation_request_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("beat_generation_requests.id", ondelete="CASCADE"),
        nullable=False
    )

    # Variation metadata
    variation_index: Mapped[int] = mapped_column(Integer, nullable=False)  # 1, 2, 3, etc.
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    # Audio files
    audio_path: Mapped[str] = mapped_column(Text, nullable=False)
    midi_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Quality and metadata
    quality_score: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(4, 2), nullable=True)
    user_rating: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-5 stars
    generation_seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # for reproducibility
    generation_metadata: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Selection and usage
    is_selected: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)  # user's preferred variation
    used_in_project: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP,
        nullable=False,
        server_default=func.now()
    )

    # Relationships
    beat_generation_request: Mapped["BeatGenerationRequest"] = relationship("BeatGenerationRequest")

    def __repr__(self) -> str:
        return f"<BeatVariation(id={self.id}, request_id={self.beat_generation_request_id}, variation={self.variation_index})>"


# Additional indexes for performance
Index("ix_projects_user_id", Project.user_id)
Index("ix_projects_created_at", Project.created_at)
Index("ix_tracks_project_id", Track.project_id)
Index("ix_tracks_project_index", Track.project_id, Track.track_index)
Index("ix_clips_track_id", Clip.track_id)
Index("ix_clips_timeline", Clip.track_id, Clip.start_time)
Index("ix_stem_separations_clip_id", StemSeparation.clip_id)
Index("ix_recording_sessions_track_id", RecordingSession.track_id)
Index("ix_recording_sessions_status", RecordingSession.status)
Index("ix_loop_regions_project_id", LoopRegion.project_id)
Index("ix_loop_regions_timeline", LoopRegion.project_id, LoopRegion.start_time, LoopRegion.end_time)

# Advanced separation and enhancement indexes
Index("ix_advanced_separations_clip_id", AdvancedSeparation.clip_id)
Index("ix_advanced_separations_type", AdvancedSeparation.separation_type)
Index("ix_advanced_separations_created", AdvancedSeparation.created_at)
Index("ix_enhancement_profiles_user_id", EnhancementProfile.user_id)
Index("ix_enhancement_profiles_default", EnhancementProfile.user_id, EnhancementProfile.is_default)
Index("ix_enhancement_profiles_public", EnhancementProfile.is_public)
Index("ix_processing_jobs_status", ProcessingJob.status)
Index("ix_processing_jobs_user", ProcessingJob.user_id, ProcessingJob.created_at)
Index("ix_processing_jobs_clip", ProcessingJob.clip_id)

# Beat generation indexes
Index("ix_beat_generation_requests_project_id", BeatGenerationRequest.project_id)
Index("ix_beat_generation_requests_user_id", BeatGenerationRequest.user_id)
Index("ix_beat_generation_requests_status", BeatGenerationRequest.status)
Index("ix_beat_generation_requests_provider", BeatGenerationRequest.provider)
Index("ix_beat_generation_requests_created", BeatGenerationRequest.created_at)
Index("ix_beat_templates_category", BeatTemplate.category)
Index("ix_beat_templates_public", BeatTemplate.is_public)
Index("ix_beat_templates_usage", BeatTemplate.usage_count)
Index("ix_beat_variations_request_id", BeatVariation.beat_generation_request_id)
Index("ix_beat_variations_selected", BeatVariation.is_selected)