"""
AudioLab Configuration Management
Centralized settings using Pydantic with environment variable support
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class AudioLabSettings(BaseSettings):
    """AudioLab application settings with environment variable support"""

    # ============================================================================
    # APPLICATION SETTINGS
    # ============================================================================
    APP_NAME: str = "AudioLab"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")

    # ============================================================================
    # DATABASE SETTINGS
    # ============================================================================
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://audiolab_user:audiolab_password@localhost:5432/audiolab",
        env="DATABASE_URL"
    )
    REDIS_URL: str = Field(
        default="redis://localhost:6379",
        env="REDIS_URL"
    )
    DATABASE_POOL_SIZE: int = Field(default=20, env="DATABASE_POOL_SIZE")
    DATABASE_MAX_OVERFLOW: int = Field(default=0, env="DATABASE_MAX_OVERFLOW")

    # ============================================================================
    # AUDIO PROCESSING SETTINGS
    # ============================================================================
    DEFAULT_SAMPLE_RATE: int = Field(default=48000, env="DEFAULT_SAMPLE_RATE")
    DEFAULT_BUFFER_SIZE: int = Field(default=512, env="DEFAULT_BUFFER_SIZE")
    MAX_TRACKS: int = Field(default=32, env="MAX_TRACKS")
    MAX_EFFECTS_PER_TRACK: int = Field(default=8, env="MAX_EFFECTS_PER_TRACK")
    AUDIO_LATENCY_TARGET_MS: float = Field(default=10.0, env="AUDIO_LATENCY_TARGET_MS")

    # Supported sample rates and bit depths
    SUPPORTED_SAMPLE_RATES: List[int] = [44100, 48000, 96000, 192000]
    SUPPORTED_BIT_DEPTHS: List[int] = [16, 24, 32]

    # ============================================================================
    # AI MODEL SETTINGS
    # ============================================================================
    DEMUCS_MODEL_PATH: str = Field(default="./models/", env="DEMUCS_MODEL_PATH")
    DEMUCS_DEFAULT_MODEL: str = Field(default="htdemucs_ft", env="DEMUCS_DEFAULT_MODEL")
    DEMUCS_SEGMENT_SIZE: int = Field(default=10, env="DEMUCS_SEGMENT_SIZE")  # seconds
    DEMUCS_USE_GPU: bool = Field(default=True, env="DEMUCS_USE_GPU")
    DEMUCS_MODELS_AVAILABLE: List[str] = [
        "htdemucs_ft",      # 4-stem: vocals, drums, bass, other
        "htdemucs_6s",      # 6-stem: vocals, drums, bass, piano, guitar, other
        "mdx_extra"         # Alternative model
    ]

    # ============================================================================
    # FILE STORAGE SETTINGS
    # ============================================================================
    AUDIO_FILES_PATH: str = Field(default="./audiolab_data/audio", env="AUDIO_FILES_PATH")
    PROJECTS_PATH: str = Field(default="./audiolab_data/projects", env="PROJECTS_PATH")
    EXPORTS_PATH: str = Field(default="./audiolab_data/exports", env="EXPORTS_PATH")
    TEMP_PATH: str = Field(default="./audiolab_data/temp", env="TEMP_PATH")
    SERVE_AUDIO_FILES: bool = Field(default=True, env="SERVE_AUDIO_FILES")

    # Maximum file sizes (in bytes)
    MAX_AUDIO_FILE_SIZE: int = Field(default=500 * 1024 * 1024, env="MAX_AUDIO_FILE_SIZE")  # 500MB
    MAX_PROJECT_SIZE: int = Field(default=2 * 1024 * 1024 * 1024, env="MAX_PROJECT_SIZE")  # 2GB

    # ============================================================================
    # SECURITY SETTINGS
    # ============================================================================
    SECRET_KEY: str = Field(
        default="audiolab-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60 * 24 * 7, env="ACCESS_TOKEN_EXPIRE_MINUTES")  # 7 days
    ALGORITHM: str = "HS256"

    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:1420",    # Tauri dev
        "tauri://localhost",        # Tauri production
        "http://localhost:3000",    # React dev server
    ]

    # ============================================================================
    # WEBSOCKET SETTINGS
    # ============================================================================
    WS_MAX_CONNECTIONS: int = Field(default=100, env="WS_MAX_CONNECTIONS")
    WS_CHUNK_SIZE: int = Field(default=4096, env="WS_CHUNK_SIZE")
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")  # seconds

    # ============================================================================
    # PERFORMANCE SETTINGS
    # ============================================================================
    WORKER_PROCESSES: int = Field(default=4, env="WORKER_PROCESSES")
    MAX_CONCURRENT_PROCESSING: int = Field(default=4, env="MAX_CONCURRENT_PROCESSING")
    CACHE_TTL_SECONDS: int = Field(default=3600, env="CACHE_TTL_SECONDS")  # 1 hour

    # Memory limits
    MAX_MEMORY_USAGE_MB: int = Field(default=2048, env="MAX_MEMORY_USAGE_MB")  # 2GB
    AUDIO_BUFFER_SIZE_MB: int = Field(default=256, env="AUDIO_BUFFER_SIZE_MB")  # 256MB

    # ============================================================================
    # EXPORT SETTINGS
    # ============================================================================
    DEFAULT_EXPORT_FORMAT: str = Field(default="wav", env="DEFAULT_EXPORT_FORMAT")
    DEFAULT_EXPORT_QUALITY: str = Field(default="24bit", env="DEFAULT_EXPORT_QUALITY")
    LUFS_TARGETS: dict = {
        "streaming": -14.0,
        "mastering": -9.0,
        "broadcast": -16.0,
        "archive": -18.0
    }

    SUPPORTED_EXPORT_FORMATS: List[str] = ["wav", "flac", "mp3", "aac"]
    SUPPORTED_EXPORT_QUALITIES: dict = {
        "wav": ["16bit", "24bit", "32bit"],
        "flac": ["16bit", "24bit"],
        "mp3": ["320kbps", "256kbps", "192kbps", "128kbps"],
        "aac": ["320kbps", "256kbps", "192kbps", "128kbps"]
    }

    # ============================================================================
    # LOGGING SETTINGS
    # ============================================================================
    LOG_FILE_PATH: str = Field(default="./logs/audiolab.log", env="LOG_FILE_PATH")
    LOG_MAX_SIZE: int = Field(default=50 * 1024 * 1024, env="LOG_MAX_SIZE")  # 50MB
    LOG_BACKUP_COUNT: int = Field(default=5, env="LOG_BACKUP_COUNT")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # ============================================================================
    # DEVELOPMENT SETTINGS
    # ============================================================================
    ENABLE_PROFILING: bool = Field(default=False, env="ENABLE_PROFILING")
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    API_RATE_LIMIT: str = Field(default="100/minute", env="API_RATE_LIMIT")

    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories"""
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        directories = [
            self.AUDIO_FILES_PATH,
            self.PROJECTS_PATH,
            self.EXPORTS_PATH,
            self.TEMP_PATH,
            self.DEMUCS_MODEL_PATH,
            Path(self.LOG_FILE_PATH).parent,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL (for migrations)"""
        return self.DATABASE_URL.replace("+asyncpg", "")

    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.DEBUG

    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return not self.DEBUG

    def get_audio_config(self) -> dict:
        """Get audio configuration dictionary"""
        return {
            "sample_rate": self.DEFAULT_SAMPLE_RATE,
            "buffer_size": self.DEFAULT_BUFFER_SIZE,
            "channels": 2,
            "latency_target_ms": self.AUDIO_LATENCY_TARGET_MS,
            "max_tracks": self.MAX_TRACKS,
        }

    def get_demucs_config(self) -> dict:
        """Get Demucs configuration dictionary"""
        return {
            "model_path": self.DEMUCS_MODEL_PATH,
            "default_model": self.DEMUCS_DEFAULT_MODEL,
            "segment_size": self.DEMUCS_SEGMENT_SIZE,
            "use_gpu": self.DEMUCS_USE_GPU,
            "available_models": self.DEMUCS_MODELS_AVAILABLE,
        }

    def get_export_config(self) -> dict:
        """Get export configuration dictionary"""
        return {
            "default_format": self.DEFAULT_EXPORT_FORMAT,
            "default_quality": self.DEFAULT_EXPORT_QUALITY,
            "lufs_targets": self.LUFS_TARGETS,
            "supported_formats": self.SUPPORTED_EXPORT_FORMATS,
            "supported_qualities": self.SUPPORTED_EXPORT_QUALITIES,
        }

    def validate_sample_rate(self, sample_rate: int) -> bool:
        """Validate sample rate against supported values"""
        return sample_rate in self.SUPPORTED_SAMPLE_RATES

    def validate_bit_depth(self, bit_depth: int) -> bool:
        """Validate bit depth against supported values"""
        return bit_depth in self.SUPPORTED_BIT_DEPTHS

    def validate_export_format(self, format_name: str) -> bool:
        """Validate export format against supported values"""
        return format_name.lower() in self.SUPPORTED_EXPORT_FORMATS


@lru_cache()
def get_settings() -> AudioLabSettings:
    """Get application settings (cached)"""
    return AudioLabSettings()


# Create settings instance for import
settings = get_settings()

# Export commonly used values
DEFAULT_SAMPLE_RATE = settings.DEFAULT_SAMPLE_RATE
DEFAULT_BUFFER_SIZE = settings.DEFAULT_BUFFER_SIZE
DATABASE_URL = settings.DATABASE_URL
REDIS_URL = settings.REDIS_URL