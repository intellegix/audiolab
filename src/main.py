"""
AudioLab - Professional Audio Production Suite
FastAPI backend with real-time audio processing and AI stem separation
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import audio, projects, tracks, effects, exports
from api.websocket import websocket_manager, audio_stream_handler
from core.config import get_settings
from core.logging import setup_logging
from database.connection import database_manager

# Global settings
settings = get_settings()

# Setup logging
logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""

    # Startup
    logger.info("Starting AudioLab backend server...")

    try:
        # Initialize database connections
        await database_manager.initialize()
        logger.info("Database connections initialized")

        # Initialize audio processing engine
        from core.audio_engine import audio_engine
        await audio_engine.initialize(
            sample_rate=settings.DEFAULT_SAMPLE_RATE,
            buffer_size=settings.DEFAULT_BUFFER_SIZE,
            channels=2
        )
        logger.info(f"Audio engine initialized: {settings.DEFAULT_SAMPLE_RATE}Hz, {settings.DEFAULT_BUFFER_SIZE} samples")

        # Load AI models
        from services.demucs_service import DemucsService
        demucs_service = DemucsService()
        await demucs_service.load_model(settings.DEMUCS_DEFAULT_MODEL)
        logger.info(f"Demucs model loaded: {settings.DEMUCS_DEFAULT_MODEL}")

        logger.info("AudioLab backend started successfully")

    except Exception as e:
        logger.error(f"Failed to start AudioLab backend: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down AudioLab backend...")

    try:
        # Stop audio processing
        await audio_engine.stop()
        logger.info("Audio engine stopped")

        # Close database connections
        await database_manager.close()
        logger.info("Database connections closed")

        logger.info("AudioLab backend shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="AudioLab API",
    description="Professional Audio Production Suite with AI Stem Separation",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "tauri://localhost"],  # Tauri dev and production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint"""
    try:
        # Check database connectivity
        db_status = await database_manager.check_health()

        # Check audio engine status
        from core.audio_engine import audio_engine
        audio_status = audio_engine.is_initialized()

        return {
            "status": "healthy",
            "version": "0.1.0",
            "services": {
                "database": "healthy" if db_status else "unhealthy",
                "audio_engine": "healthy" if audio_status else "unhealthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


# API Routes
app.include_router(audio.router, prefix="/api/audio", tags=["Audio Processing"])
app.include_router(projects.router, prefix="/api/projects", tags=["Projects"])
app.include_router(tracks.router, prefix="/api/tracks", tags=["Tracks"])
app.include_router(effects.router, prefix="/api/effects", tags=["Effects"])
app.include_router(exports.router, prefix="/api/exports", tags=["Export"])

# WebSocket endpoints
@app.websocket("/ws/audio/{project_id}")
async def websocket_audio_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time audio processing"""

    connection_id = f"{project_id}_{hash(websocket)}"

    try:
        # Accept connection
        await websocket_manager.connect(websocket, connection_id, project_id)
        logger.info(f"WebSocket connected: {connection_id}")

        # Start audio stream
        await audio_stream_handler.start_audio_stream(
            connection_id,
            project_id,
            {
                "sample_rate": settings.DEFAULT_SAMPLE_RATE,
                "buffer_size": settings.DEFAULT_BUFFER_SIZE,
                "channels": 2
            }
        )

        # Message handling loop
        while True:
            try:
                # Receive message
                message = await websocket.receive_text()

                # Process message
                await audio_stream_handler.handle_message(connection_id, message)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                await websocket_manager.send_error(connection_id, str(e))
                break

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")

    finally:
        # Clean up
        await audio_stream_handler.stop_audio_stream(connection_id)
        await websocket_manager.disconnect(connection_id)
        logger.info(f"WebSocket disconnected: {connection_id}")


@app.websocket("/ws/collaboration/{project_id}")
async def websocket_collaboration_endpoint(websocket: WebSocket, project_id: str):
    """WebSocket endpoint for real-time collaboration"""

    connection_id = f"collab_{project_id}_{hash(websocket)}"

    try:
        await websocket_manager.connect(websocket, connection_id, project_id)

        # Collaboration message loop
        while True:
            message = await websocket.receive_text()
            # Broadcast to all project collaborators
            await websocket_manager.broadcast_to_project(project_id, message)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Collaboration WebSocket error: {e}")
    finally:
        await websocket_manager.disconnect(connection_id)


# Static file serving (for audio files)
if settings.SERVE_AUDIO_FILES:
    app.mount("/audio", StaticFiles(directory=settings.AUDIO_FILES_PATH), name="audio")


# Application info
@app.get("/api/info")
async def app_info() -> Dict[str, Any]:
    """Get application information"""
    return {
        "name": "AudioLab",
        "version": "0.1.0",
        "description": "Professional Audio Production Suite",
        "author": "Austin Kidwell | Intellegix, ASR Inc",
        "features": [
            "AI Stem Separation (Demucs v4)",
            "Real-time Audio Processing",
            "Multi-track Mixing",
            "Cross-platform Desktop",
            "WebSocket Streaming",
            "Professional Effects"
        ],
        "tech_stack": {
            "backend": "FastAPI + Python",
            "database": "PostgreSQL + Redis",
            "ai_models": "Demucs v4, PyTorch",
            "audio": "Librosa, SoundFile, PyAudio"
        }
    }


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )