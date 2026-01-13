"""Add overdubbing and looping schema

Revision ID: 001_add_overdubbing_schema
Revises: eed95b4717bc
Create Date: 2026-01-12 15:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_add_overdubbing_schema'
down_revision: Union[str, None] = 'eed95b4717bc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add overdubbing and looping schema"""

    # Add recording fields to tracks table
    op.add_column('tracks', sa.Column('record_enabled', sa.Boolean(), nullable=False, server_default='false'))
    op.add_column('tracks', sa.Column('input_device_id', sa.String(255), nullable=True))
    op.add_column('tracks', sa.Column('monitoring_enabled', sa.Boolean(), nullable=False, server_default='true'))

    # Create recording_sessions table
    op.create_table('recording_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('track_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tracks.id', ondelete='CASCADE'), nullable=False),
        sa.Column('input_device_id', sa.String(255), nullable=False),
        sa.Column('start_time', sa.DECIMAL(10, 6), nullable=False),
        sa.Column('duration', sa.DECIMAL(10, 6), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='recording'),
        sa.Column('sample_rate', sa.Integer(), nullable=False, server_default='48000'),
        sa.Column('channels', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('bit_depth', sa.Integer(), nullable=False, server_default='24'),
        sa.Column('temp_file_path', sa.String(500), nullable=True),
        sa.Column('final_clip_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('clips.id', ondelete='SET NULL'), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Create loop_regions table
    op.create_table('loop_regions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False, server_default='Loop Region'),
        sa.Column('start_time', sa.DECIMAL(10, 6), nullable=False),
        sa.Column('end_time', sa.DECIMAL(10, 6), nullable=False),
        sa.Column('is_enabled', sa.Boolean(), nullable=False, server_default='true'),
        sa.Column('repeat_count', sa.Integer(), nullable=True),
        sa.Column('auto_punch_record', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('color', sa.String(7), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now())
    )

    # Create indexes for performance
    op.create_index('ix_recording_sessions_track_id', 'recording_sessions', ['track_id'])
    op.create_index('ix_recording_sessions_status', 'recording_sessions', ['status'])
    op.create_index('ix_loop_regions_project_id', 'loop_regions', ['project_id'])
    op.create_index('ix_loop_regions_timeline', 'loop_regions', ['project_id', 'start_time', 'end_time'])


def downgrade() -> None:
    """Remove overdubbing and looping schema"""

    # Drop indexes
    op.drop_index('ix_loop_regions_timeline', 'loop_regions')
    op.drop_index('ix_loop_regions_project_id', 'loop_regions')
    op.drop_index('ix_recording_sessions_status', 'recording_sessions')
    op.drop_index('ix_recording_sessions_track_id', 'recording_sessions')

    # Drop tables
    op.drop_table('loop_regions')
    op.drop_table('recording_sessions')

    # Remove columns from tracks table
    op.drop_column('tracks', 'monitoring_enabled')
    op.drop_column('tracks', 'input_device_id')
    op.drop_column('tracks', 'record_enabled')