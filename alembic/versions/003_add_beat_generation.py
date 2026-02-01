"""Add beat generation tables

Revision ID: 003_add_beat_generation
Revises: 002_add_advanced_separation
Create Date: 2026-02-01 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, DECIMAL


# revision identifiers, used by Alembic.
revision: str = '003_add_beat_generation'
down_revision: Union[str, None] = '002_add_advanced_separation'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add beat generation tables and indexes"""

    # Create beat_generation_requests table
    op.create_table(
        'beat_generation_requests',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', UUID(as_uuid=True), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', UUID(as_uuid=True), nullable=False),

        # Generation parameters
        sa.Column('prompt', sa.Text, nullable=False),
        sa.Column('provider', sa.String(20), nullable=False, server_default='musicgen'),
        sa.Column('model_name', sa.String(100), nullable=True),
        sa.Column('duration', DECIMAL(6, 2), nullable=False),

        # Musical synchronization
        sa.Column('tempo', DECIMAL(5, 2), nullable=False),
        sa.Column('time_signature', sa.String(10), nullable=False),
        sa.Column('style_tags', JSONB, nullable=True),

        # Processing state
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('progress', DECIMAL(5, 2), nullable=False, server_default='0.0'),
        sa.Column('current_stage', sa.String(100), nullable=True),

        # Results
        sa.Column('generated_audio_path', sa.Text, nullable=True),
        sa.Column('generated_midi_path', sa.Text, nullable=True),
        sa.Column('quality_score', DECIMAL(4, 2), nullable=True),
        sa.Column('processing_time', DECIMAL(8, 3), nullable=True),
        sa.Column('provider_metadata', JSONB, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),

        # Timestamps
        sa.Column('created_at', sa.TIMESTAMP, nullable=False, server_default=sa.func.now()),
        sa.Column('started_at', sa.TIMESTAMP, nullable=True),
        sa.Column('completed_at', sa.TIMESTAMP, nullable=True),
    )

    # Create beat_templates table
    op.create_table(
        'beat_templates',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),

        # Template metadata
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('category', sa.String(50), nullable=False),
        sa.Column('tags', JSONB, nullable=False),

        # Musical parameters
        sa.Column('default_tempo', DECIMAL(5, 2), nullable=False),
        sa.Column('time_signature', sa.String(10), nullable=False),
        sa.Column('duration', DECIMAL(6, 2), nullable=False),

        # Generation parameters
        sa.Column('provider_config', JSONB, nullable=False),
        sa.Column('prompt_template', sa.Text, nullable=False),

        # Usage and quality metrics
        sa.Column('usage_count', sa.Integer, nullable=False, server_default='0'),
        sa.Column('average_quality', DECIMAL(4, 2), nullable=True),
        sa.Column('is_public', sa.Boolean, nullable=False, server_default='true'),
        sa.Column('created_by_user_id', UUID(as_uuid=True), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.TIMESTAMP, nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP, nullable=False, server_default=sa.func.now()),
    )

    # Create beat_variations table
    op.create_table(
        'beat_variations',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('beat_generation_request_id', UUID(as_uuid=True), sa.ForeignKey('beat_generation_requests.id', ondelete='CASCADE'), nullable=False),

        # Variation metadata
        sa.Column('variation_index', sa.Integer, nullable=False),
        sa.Column('name', sa.String(255), nullable=True),

        # Audio files
        sa.Column('audio_path', sa.Text, nullable=False),
        sa.Column('midi_path', sa.Text, nullable=True),

        # Quality and metadata
        sa.Column('quality_score', DECIMAL(4, 2), nullable=True),
        sa.Column('user_rating', sa.Integer, nullable=True),
        sa.Column('generation_seed', sa.Integer, nullable=True),
        sa.Column('generation_metadata', JSONB, nullable=True),

        # Selection and usage
        sa.Column('is_selected', sa.Boolean, nullable=False, server_default='false'),
        sa.Column('used_in_project', sa.Boolean, nullable=False, server_default='false'),

        # Timestamps
        sa.Column('created_at', sa.TIMESTAMP, nullable=False, server_default=sa.func.now()),
    )

    # Create indexes for beat generation tables
    op.create_index('ix_beat_generation_requests_project_id', 'beat_generation_requests', ['project_id'])
    op.create_index('ix_beat_generation_requests_user_id', 'beat_generation_requests', ['user_id'])
    op.create_index('ix_beat_generation_requests_status', 'beat_generation_requests', ['status'])
    op.create_index('ix_beat_generation_requests_provider', 'beat_generation_requests', ['provider'])
    op.create_index('ix_beat_generation_requests_created', 'beat_generation_requests', ['created_at'])

    op.create_index('ix_beat_templates_category', 'beat_templates', ['category'])
    op.create_index('ix_beat_templates_public', 'beat_templates', ['is_public'])
    op.create_index('ix_beat_templates_usage', 'beat_templates', ['usage_count'])

    op.create_index('ix_beat_variations_request_id', 'beat_variations', ['beat_generation_request_id'])
    op.create_index('ix_beat_variations_selected', 'beat_variations', ['is_selected'])

    # Update ProcessingJob job_type comment to include beat_generation
    op.execute("""
        COMMENT ON COLUMN processing_jobs.job_type IS
        'Type of processing job: advanced_separation, enhancement, stem_separation, beat_generation'
    """)


def downgrade() -> None:
    """Remove beat generation tables and indexes"""

    # Drop indexes
    op.drop_index('ix_beat_variations_selected', 'beat_variations')
    op.drop_index('ix_beat_variations_request_id', 'beat_variations')
    op.drop_index('ix_beat_templates_usage', 'beat_templates')
    op.drop_index('ix_beat_templates_public', 'beat_templates')
    op.drop_index('ix_beat_templates_category', 'beat_templates')
    op.drop_index('ix_beat_generation_requests_created', 'beat_generation_requests')
    op.drop_index('ix_beat_generation_requests_provider', 'beat_generation_requests')
    op.drop_index('ix_beat_generation_requests_status', 'beat_generation_requests')
    op.drop_index('ix_beat_generation_requests_user_id', 'beat_generation_requests')
    op.drop_index('ix_beat_generation_requests_project_id', 'beat_generation_requests')

    # Drop tables (order matters due to foreign keys)
    op.drop_table('beat_variations')
    op.drop_table('beat_templates')
    op.drop_table('beat_generation_requests')

    # Revert ProcessingJob comment
    op.execute("""
        COMMENT ON COLUMN processing_jobs.job_type IS
        'Type of processing job: advanced_separation, enhancement, stem_separation'
    """)