"""Initial AudioLab schema

Revision ID: eed95b4717bc
Revises: 
Create Date: 2026-01-12 14:19:49.887649

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB


# revision identifiers, used by Alembic.
revision: str = 'eed95b4717bc'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create projects table
    op.create_table(
        'projects',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('sample_rate', sa.Integer, nullable=False, default=48000),
        sa.Column('bit_depth', sa.Integer, nullable=False, default=24),
        sa.Column('tempo', sa.DECIMAL(5, 2), nullable=False, default=120.0),
        sa.Column('time_signature', sa.String(10), nullable=False, default='4/4'),
        sa.Column('user_id', UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP, nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.TIMESTAMP, nullable=False, server_default=sa.text('now()'))
    )

    # Create tracks table
    op.create_table(
        'tracks',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('track_index', sa.Integer, nullable=False),
        sa.Column('volume', sa.DECIMAL(3, 2), nullable=False, default=1.0),
        sa.Column('pan', sa.DECIMAL(3, 2), nullable=False, default=0.0),
        sa.Column('muted', sa.Boolean, nullable=False, default=False),
        sa.Column('soloed', sa.Boolean, nullable=False, default=False),
        sa.Column('color', sa.String(7), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP, nullable=False, server_default=sa.text('now()'))
    )

    # Create clips table
    op.create_table(
        'clips',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('track_id', UUID(as_uuid=True), sa.ForeignKey('tracks.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('file_path', sa.Text, nullable=False),
        sa.Column('start_time', sa.DECIMAL(10, 6), nullable=False),
        sa.Column('duration', sa.DECIMAL(10, 6), nullable=False),
        sa.Column('offset', sa.DECIMAL(10, 6), nullable=False, default=0.0),
        sa.Column('fade_in', sa.DECIMAL(6, 3), nullable=False, default=0.0),
        sa.Column('fade_out', sa.DECIMAL(6, 3), nullable=False, default=0.0),
        sa.Column('gain', sa.DECIMAL(6, 2), nullable=False, default=0.0),
        sa.Column('created_at', sa.TIMESTAMP, nullable=False, server_default=sa.text('now()'))
    )

    # Create effects table
    op.create_table(
        'effects',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('track_id', UUID(as_uuid=True), sa.ForeignKey('tracks.id', ondelete='CASCADE'), nullable=True),
        sa.Column('clip_id', UUID(as_uuid=True), sa.ForeignKey('clips.id', ondelete='CASCADE'), nullable=True),
        sa.Column('name', sa.String(100), nullable=False),
        sa.Column('effect_type', sa.String(50), nullable=False),
        sa.Column('parameters', JSONB, nullable=False),
        sa.Column('bypass', sa.Boolean, nullable=False, default=False),
        sa.Column('order_index', sa.Integer, nullable=False),
        sa.Column('created_at', sa.TIMESTAMP, nullable=False, server_default=sa.text('now()'))
    )

    # Create stem_separations table
    op.create_table(
        'stem_separations',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('clip_id', UUID(as_uuid=True), sa.ForeignKey('clips.id', ondelete='CASCADE'), nullable=False),
        sa.Column('stems', JSONB, nullable=False),
        sa.Column('model_used', sa.String(50), nullable=False, default='htdemucs_ft'),
        sa.Column('processing_time', sa.DECIMAL(8, 3), nullable=True),
        sa.Column('quality_score', sa.DECIMAL(4, 2), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP, nullable=False, server_default=sa.text('now()'))
    )

    # Create indexes for performance
    op.create_index('ix_projects_user_id', 'projects', ['user_id'])
    op.create_index('ix_projects_created_at', 'projects', ['created_at'])
    op.create_index('ix_tracks_project_id', 'tracks', ['project_id'])
    op.create_index('ix_tracks_project_index', 'tracks', ['project_id', 'track_index'])
    op.create_index('ix_clips_track_id', 'clips', ['track_id'])
    op.create_index('ix_clips_timeline', 'clips', ['track_id', 'start_time'])
    op.create_index('ix_effect_track_id', 'effects', ['track_id'])
    op.create_index('ix_effect_clip_id', 'effects', ['clip_id'])
    op.create_index('ix_effect_order', 'effects', ['track_id', 'clip_id', 'order_index'])
    op.create_index('ix_stem_separations_clip_id', 'stem_separations', ['clip_id'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_stem_separations_clip_id')
    op.drop_index('ix_effect_order')
    op.drop_index('ix_effect_clip_id')
    op.drop_index('ix_effect_track_id')
    op.drop_index('ix_clips_timeline')
    op.drop_index('ix_clips_track_id')
    op.drop_index('ix_tracks_project_index')
    op.drop_index('ix_tracks_project_id')
    op.drop_index('ix_projects_created_at')
    op.drop_index('ix_projects_user_id')

    # Drop tables (order matters due to foreign keys)
    op.drop_table('stem_separations')
    op.drop_table('effects')
    op.drop_table('clips')
    op.drop_table('tracks')
    op.drop_table('projects')
