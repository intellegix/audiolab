"""
AudioLab WebSocket Management
Real-time audio streaming and collaboration via WebSockets
"""

from typing import Dict, Set
import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect

from ..core.logging import websocket_logger


class ConnectionManager:
    """Manages WebSocket connections for real-time audio"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.project_connections: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, connection_id: str, project_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()

        async with self._lock:
            self.active_connections[connection_id] = websocket

            if project_id not in self.project_connections:
                self.project_connections[project_id] = set()
            self.project_connections[project_id].add(connection_id)

        websocket_logger.log_connection(connection_id, project_id)

    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        async with self._lock:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

            # Remove from project connections
            for project_id, connections in self.project_connections.items():
                connections.discard(connection_id)

        websocket_logger.log_disconnection(connection_id)

    async def send_message(self, connection_id: str, message: dict):
        """Send message to specific connection"""
        if connection_id not in self.active_connections:
            return False

        websocket = self.active_connections[connection_id]

        try:
            await websocket.send_text(json.dumps(message))
            websocket_logger.log_message_sent(
                connection_id,
                message.get("type", "unknown"),
                len(json.dumps(message))
            )
            return True

        except WebSocketDisconnect:
            await self.disconnect(connection_id)
            return False

    async def send_error(self, connection_id: str, error: str):
        """Send error message to connection"""
        await self.send_message(connection_id, {
            "type": "error",
            "data": {"error": error}
        })

    async def broadcast_to_project(self, project_id: str, message: dict):
        """Broadcast message to all connections in a project"""
        if project_id not in self.project_connections:
            return

        connections = self.project_connections[project_id].copy()
        failed_connections = []

        for connection_id in connections:
            success = await self.send_message(connection_id, message)
            if not success:
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)

        websocket_logger.log_broadcast(
            project_id,
            message.get("type", "unknown"),
            len(connections) - len(failed_connections)
        )

    async def send_progress_update(
        self,
        connection_id: str,
        operation: str,
        progress: float,
        message: str,
        metadata: dict = None
    ):
        """Send processing progress update to connection"""
        progress_message = {
            "type": "progress",
            "data": {
                "operation": operation,
                "progress": min(max(progress, 0.0), 1.0),  # Clamp between 0-1
                "message": message,
                "metadata": metadata or {}
            }
        }
        await self.send_message(connection_id, progress_message)

    async def send_processing_complete(
        self,
        connection_id: str,
        operation: str,
        result: dict,
        duration_ms: float = None
    ):
        """Send processing completion notification"""
        completion_message = {
            "type": "processing_complete",
            "data": {
                "operation": operation,
                "result": result,
                "duration_ms": duration_ms,
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        await self.send_message(connection_id, completion_message)

    async def send_processing_error(
        self,
        connection_id: str,
        operation: str,
        error: str,
        error_code: str = None
    ):
        """Send processing error notification"""
        error_message = {
            "type": "processing_error",
            "data": {
                "operation": operation,
                "error": error,
                "error_code": error_code,
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        await self.send_message(connection_id, error_message)

    def create_progress_callback(self, connection_id: str, operation: str):
        """Create a progress callback function for Demucs processing"""
        async def progress_callback(progress: float, message: str):
            await self.send_progress_update(
                connection_id,
                operation,
                progress,
                message
            )

        return progress_callback


class AudioStreamHandler:
    """Handles real-time audio streaming via WebSocket"""

    def __init__(self):
        self.active_streams: Dict[str, dict] = {}

    async def start_audio_stream(
        self,
        connection_id: str,
        project_id: str,
        config: dict
    ):
        """Start audio processing stream for connection"""
        self.active_streams[connection_id] = {
            "project_id": project_id,
            "config": config,
            "started_at": asyncio.get_event_loop().time()
        }

    async def stop_audio_stream(self, connection_id: str):
        """Stop audio processing stream"""
        if connection_id in self.active_streams:
            del self.active_streams[connection_id]

    async def handle_message(self, connection_id: str, message_str: str):
        """Handle incoming WebSocket message"""
        try:
            message = json.loads(message_str)
            message_type = message.get("type")

            websocket_logger.log_message_received(
                connection_id,
                message_type,
                len(message_str)
            )

            if message_type == "audio_data":
                await self._handle_audio_data(connection_id, message)
            elif message_type == "control":
                await self._handle_control_message(connection_id, message)
            elif message_type == "ping":
                await websocket_manager.send_message(connection_id, {
                    "type": "pong"
                })

        except Exception as e:
            await websocket_manager.send_error(connection_id, str(e))

    async def _handle_audio_data(self, connection_id: str, message: dict):
        """Handle incoming audio data"""
        # TODO: Implement audio data processing
        pass

    async def _handle_control_message(self, connection_id: str, message: dict):
        """Handle control messages (play, stop, etc.)"""
        # TODO: Implement control message handling
        pass


# Global instances
websocket_manager = ConnectionManager()
audio_stream_handler = AudioStreamHandler()