"""
AudioLab Alembic Environment Configuration
Configured for async SQLAlchemy and AudioLab models
"""

import asyncio
import sys
import os
from logging.config import fileConfig
from sqlalchemy import pool, MetaData
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base
from alembic import context

# Add src to path for model imports
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

# Create Base and import models directly
Base = declarative_base()

# Import models with absolute imports after setting path
try:
    # Import AudioLab models
    import uuid
    from datetime import datetime
    from decimal import Decimal
    from sqlalchemy import Column, String, Integer, Boolean, Text, TIMESTAMP, ForeignKey, Index, DECIMAL
    from sqlalchemy.dialects.postgresql import UUID, JSONB
    from sqlalchemy.sql import func
    from sqlalchemy.orm import relationship, Mapped, mapped_column

    # Define models inline to avoid import issues
    class Project(Base):
        __tablename__ = "projects"
        id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid())
        name: Mapped[str] = mapped_column(String(255), nullable=False)
        sample_rate: Mapped[int] = mapped_column(Integer, default=48000, nullable=False)
        bit_depth: Mapped[int] = mapped_column(Integer, default=24, nullable=False)
        tempo: Mapped[Decimal] = mapped_column(DECIMAL(5, 2), default=Decimal("120.0"), nullable=False)
        time_signature: Mapped[str] = mapped_column(String(10), default="4/4", nullable=False)
        user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
        created_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())
        updated_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now(), onupdate=func.now())

    class Track(Base):
        __tablename__ = "tracks"
        id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid())
        project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False)
        name: Mapped[str] = mapped_column(String(255), nullable=False)
        track_index: Mapped[int] = mapped_column(Integer, nullable=False)
        volume: Mapped[Decimal] = mapped_column(DECIMAL(3, 2), default=Decimal("1.0"), nullable=False)
        pan: Mapped[Decimal] = mapped_column(DECIMAL(3, 2), default=Decimal("0.0"), nullable=False)
        muted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
        soloed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
        color: Mapped[str] = mapped_column(String(7), nullable=True)
        created_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())

    class Clip(Base):
        __tablename__ = "clips"
        id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid())
        track_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False)
        name: Mapped[str] = mapped_column(String(255), nullable=True)
        file_path: Mapped[str] = mapped_column(Text, nullable=False)
        start_time: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)
        duration: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)
        offset: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), default=Decimal("0.0"), nullable=False)
        fade_in: Mapped[Decimal] = mapped_column(DECIMAL(6, 3), default=Decimal("0.0"), nullable=False)
        fade_out: Mapped[Decimal] = mapped_column(DECIMAL(6, 3), default=Decimal("0.0"), nullable=False)
        gain: Mapped[Decimal] = mapped_column(DECIMAL(6, 2), default=Decimal("0.0"), nullable=False)
        created_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())

    class Effect(Base):
        __tablename__ = "effects"
        id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid())
        track_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tracks.id", ondelete="CASCADE"), nullable=True)
        clip_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("clips.id", ondelete="CASCADE"), nullable=True)
        name: Mapped[str] = mapped_column(String(100), nullable=False)
        effect_type: Mapped[str] = mapped_column(String(50), nullable=False)
        parameters: Mapped[dict] = mapped_column(JSONB, nullable=False)
        bypass: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
        order_index: Mapped[int] = mapped_column(Integer, nullable=False)
        created_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())

    class StemSeparation(Base):
        __tablename__ = "stem_separations"
        id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, server_default=func.gen_random_uuid())
        clip_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("clips.id", ondelete="CASCADE"), nullable=False)
        stems: Mapped[dict] = mapped_column(JSONB, nullable=False)
        model_used: Mapped[str] = mapped_column(String(50), default="htdemucs_ft", nullable=False)
        processing_time: Mapped[Decimal] = mapped_column(DECIMAL(8, 3), nullable=True)
        quality_score: Mapped[Decimal] = mapped_column(DECIMAL(4, 2), nullable=True)
        created_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False, server_default=func.now())

    # Create indexes
    Index("ix_projects_user_id", Project.user_id)
    Index("ix_projects_created_at", Project.created_at)
    Index("ix_tracks_project_id", Track.project_id)
    Index("ix_tracks_project_index", Track.project_id, Track.track_index)
    Index("ix_clips_track_id", Clip.track_id)
    Index("ix_clips_timeline", Clip.track_id, Clip.start_time)
    Index("ix_effect_track_id", Effect.track_id)
    Index("ix_effect_clip_id", Effect.clip_id)
    Index("ix_effect_order", Effect.track_id, Effect.clip_id, Effect.order_index)
    Index("ix_stem_separations_clip_id", StemSeparation.clip_id)

except Exception as e:
    print(f"Error importing models: {e}")
    raise

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set default database URL (will be overridden by AudioLab config if available)
default_url = "postgresql://audiolab_user:audiolab_password@localhost:5432/audiolab"
config.set_main_option("sqlalchemy.url", default_url)

# Add model metadata for autogenerate support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with the given connection"""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in async mode"""
    url = config.get_main_option("sqlalchemy.url")

    # Use async engine for migrations
    connectable = create_async_engine(
        url.replace("postgresql://", "postgresql+asyncpg://"),
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
