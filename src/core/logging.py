"""
AudioLab Logging Configuration
Structured logging setup with file rotation and performance monitoring
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Any, Dict

import structlog
from pythonjsonlogger import jsonlogger

from .config import get_settings

settings = get_settings()


def setup_logging() -> logging.Logger:
    """Set up structured logging for AudioLab"""

    # Create logs directory
    log_dir = Path(settings.LOG_FILE_PATH).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure standard library logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper()),
        format=settings.LOG_FORMAT,
        handlers=[]  # Will be set below
    )

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Console handler with colored output for development
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    if settings.is_development:
        # Colored console formatter for development
        console_formatter = logging.Formatter(
            '\033[92m%(asctime)s\033[0m - '
            '\033[94m%(name)s\033[0m - '
            '\033[%(levelno)s;1m%(levelname)s\033[0m - '
            '%(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        # JSON formatter for production
        console_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        settings.LOG_FILE_PATH,
        maxBytes=settings.LOG_MAX_SIZE,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)

    # JSON formatter for log files
    file_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(pathname)s %(lineno)d %(funcName)s %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Configure structlog for structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if not settings.is_development
            else structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Adjust third-party library log levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Audio processing specific loggers
    logging.getLogger("audiolab.audio").setLevel(logging.DEBUG)
    logging.getLogger("audiolab.demucs").setLevel(logging.INFO)
    logging.getLogger("audiolab.websocket").setLevel(logging.INFO)

    # Get configured logger
    logger = logging.getLogger("audiolab")
    logger.info(f"Logging configured - Level: {settings.LOG_LEVEL}")

    return logger


class AudioProcessingLogger:
    """Specialized logger for audio processing operations"""

    def __init__(self):
        self.logger = structlog.get_logger("audiolab.audio")

    def log_processing_start(
        self,
        operation: str,
        file_path: str = None,
        **kwargs: Any
    ) -> None:
        """Log start of audio processing operation"""
        self.logger.info(
            "Audio processing started",
            operation=operation,
            file_path=file_path,
            **kwargs
        )

    def log_processing_complete(
        self,
        operation: str,
        duration_ms: float,
        file_path: str = None,
        **kwargs: Any
    ) -> None:
        """Log completion of audio processing operation"""
        self.logger.info(
            "Audio processing completed",
            operation=operation,
            duration_ms=duration_ms,
            file_path=file_path,
            **kwargs
        )

    def log_processing_error(
        self,
        operation: str,
        error: str,
        file_path: str = None,
        **kwargs: Any
    ) -> None:
        """Log audio processing error"""
        self.logger.error(
            "Audio processing failed",
            operation=operation,
            error=error,
            file_path=file_path,
            **kwargs
        )

    def log_demucs_separation(
        self,
        model: str,
        duration_s: float,
        processing_time_s: float,
        stems: list,
        gpu_used: bool = False
    ) -> None:
        """Log Demucs stem separation operation"""
        self.logger.info(
            "Demucs stem separation completed",
            model=model,
            audio_duration_s=duration_s,
            processing_time_s=processing_time_s,
            performance_ratio=processing_time_s / duration_s,
            stems=stems,
            gpu_used=gpu_used
        )

    def log_realtime_stats(
        self,
        buffer_underruns: int,
        buffer_overruns: int,
        latency_ms: float,
        cpu_usage: float
    ) -> None:
        """Log real-time audio statistics"""
        self.logger.debug(
            "Real-time audio stats",
            buffer_underruns=buffer_underruns,
            buffer_overruns=buffer_overruns,
            latency_ms=latency_ms,
            cpu_usage_percent=cpu_usage
        )


class WebSocketLogger:
    """Specialized logger for WebSocket operations"""

    def __init__(self):
        self.logger = structlog.get_logger("audiolab.websocket")

    def log_connection(self, connection_id: str, project_id: str) -> None:
        """Log WebSocket connection"""
        self.logger.info(
            "WebSocket connection established",
            connection_id=connection_id,
            project_id=project_id
        )

    def log_disconnection(
        self,
        connection_id: str,
        reason: str = None
    ) -> None:
        """Log WebSocket disconnection"""
        self.logger.info(
            "WebSocket connection closed",
            connection_id=connection_id,
            reason=reason
        )

    def log_message_received(
        self,
        connection_id: str,
        message_type: str,
        size_bytes: int
    ) -> None:
        """Log WebSocket message received"""
        self.logger.debug(
            "WebSocket message received",
            connection_id=connection_id,
            message_type=message_type,
            size_bytes=size_bytes
        )

    def log_message_sent(
        self,
        connection_id: str,
        message_type: str,
        size_bytes: int
    ) -> None:
        """Log WebSocket message sent"""
        self.logger.debug(
            "WebSocket message sent",
            connection_id=connection_id,
            message_type=message_type,
            size_bytes=size_bytes
        )

    def log_broadcast(
        self,
        project_id: str,
        message_type: str,
        recipients: int
    ) -> None:
        """Log WebSocket broadcast"""
        self.logger.debug(
            "WebSocket broadcast sent",
            project_id=project_id,
            message_type=message_type,
            recipients=recipients
        )


class PerformanceLogger:
    """Logger for performance monitoring"""

    def __init__(self):
        self.logger = structlog.get_logger("audiolab.performance")

    def log_memory_usage(
        self,
        operation: str,
        memory_mb: float,
        peak_memory_mb: float = None
    ) -> None:
        """Log memory usage during operation"""
        self.logger.debug(
            "Memory usage",
            operation=operation,
            memory_mb=memory_mb,
            peak_memory_mb=peak_memory_mb
        )

    def log_cpu_usage(
        self,
        operation: str,
        cpu_percent: float,
        cores_used: int = None
    ) -> None:
        """Log CPU usage during operation"""
        self.logger.debug(
            "CPU usage",
            operation=operation,
            cpu_percent=cpu_percent,
            cores_used=cores_used
        )

    def log_database_query(
        self,
        query: str,
        duration_ms: float,
        rows_affected: int = None
    ) -> None:
        """Log database query performance"""
        self.logger.debug(
            "Database query",
            query=query[:100] + "..." if len(query) > 100 else query,
            duration_ms=duration_ms,
            rows_affected=rows_affected
        )


# Create global logger instances
audio_logger = AudioProcessingLogger()
websocket_logger = WebSocketLogger()
performance_logger = PerformanceLogger()

# Export for convenience
__all__ = [
    "setup_logging",
    "AudioProcessingLogger",
    "WebSocketLogger",
    "PerformanceLogger",
    "audio_logger",
    "websocket_logger",
    "performance_logger"
]