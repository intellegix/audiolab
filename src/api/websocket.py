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

    async def send_position_update(self, project_id: str, position: float):
        """Send real-time playback position update to all project connections"""
        message = {
            "type": "position_update",
            "data": {
                "project_id": project_id,
                "position": position,
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        await self.broadcast_to_project(project_id, message)

    async def send_playback_status_update(self, project_id: str, status: dict):
        """Send playback status update to all project connections"""
        message = {
            "type": "playback_status_update",
            "data": {
                "project_id": project_id,
                "status": status,
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        await self.broadcast_to_project(project_id, message)

    async def send_recording_status_update(self, connection_id: str, session_data: dict):
        """Send recording session status update"""
        message = {
            "type": "recording_status_update",
            "data": {
                **session_data,
                "timestamp": asyncio.get_event_loop().time()
            }
        }

        if connection_id:
            await self.send_message(connection_id, message)
        else:
            # Broadcast if no specific connection
            if "track_id" in session_data:
                project_id = session_data.get("project_id")
                if project_id:
                    await self.broadcast_to_project(project_id, message)

    async def send_loop_event(self, project_id: str, event_type: str, loop_data: dict):
        """Send loop event notification (loop start, end, enabled, disabled)"""
        message = {
            "type": "loop_event",
            "data": {
                "event_type": event_type,
                "loop_data": loop_data,
                "project_id": project_id,
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        await self.broadcast_to_project(project_id, message)

    async def send_track_update(self, project_id: str, track_id: str, update_data: dict):
        """Send track parameter update (volume, mute, solo, etc.)"""
        message = {
            "type": "track_update",
            "data": {
                "track_id": track_id,
                "project_id": project_id,
                "update_data": update_data,
                "timestamp": asyncio.get_event_loop().time()
            }
        }
        await self.broadcast_to_project(project_id, message)

    async def send_system_notification(self, message_text: str, level: str = "info",
                                     connection_id: str = None, project_id: str = None):
        """Send system notification to specific connection or project"""
        message = {
            "type": "system_notification",
            "data": {
                "message": message_text,
                "level": level,  # info, warning, error, success
                "timestamp": asyncio.get_event_loop().time()
            }
        }

        if connection_id:
            await self.send_message(connection_id, message)
        elif project_id:
            await self.broadcast_to_project(project_id, message)

    def create_recording_callback(self, connection_id: str = None):
        """Create a callback function for recording service"""
        async def recording_callback(session_data: dict):
            await self.send_recording_status_update(connection_id, session_data)

        return recording_callback

    def create_playback_callback(self, project_id: str):
        """Create a callback function for playback service"""
        async def playback_callback(status_data: dict):
            await self.send_playback_status_update(project_id, status_data)

        return playback_callback

    def create_loop_callback(self, project_id: str):
        """Create a callback function for loop service"""
        async def loop_callback(event_type: str, loop_data: dict):
            await self.send_loop_event(project_id, event_type, loop_data)

        return loop_callback


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
        """Handle control messages (play, stop, record, etc.)"""
        try:
            command = message.get("command")
            data = message.get("data", {})

            if command == "play":
                await self._handle_play_command(connection_id, data)
            elif command == "stop":
                await self._handle_stop_command(connection_id, data)
            elif command == "pause":
                await self._handle_pause_command(connection_id, data)
            elif command == "resume":
                await self._handle_resume_command(connection_id, data)
            elif command == "seek":
                await self._handle_seek_command(connection_id, data)
            elif command == "record_start":
                await self._handle_record_start_command(connection_id, data)
            elif command == "record_stop":
                await self._handle_record_stop_command(connection_id, data)
            elif command == "loop_enable":
                await self._handle_loop_enable_command(connection_id, data)
            elif command == "loop_disable":
                await self._handle_loop_disable_command(connection_id, data)
            else:
                await websocket_manager.send_error(connection_id, f"Unknown command: {command}")

        except Exception as e:
            await websocket_manager.send_error(connection_id, f"Control message error: {str(e)}")

    async def _handle_play_command(self, connection_id: str, data: dict):
        """Handle play command"""
        try:
            from ...services.playback_service import get_playback_service

            project_id = data.get("project_id")
            position = data.get("position", 0.0)

            if not project_id:
                await websocket_manager.send_error(connection_id, "Missing project_id")
                return

            playback_service = await get_playback_service()
            result = await playback_service.play(start_position=position, connection_id=connection_id)

            await websocket_manager.send_message(connection_id, {
                "type": "control_response",
                "data": {
                    "command": "play",
                    "success": result["success"],
                    "result": result
                }
            })

        except Exception as e:
            await websocket_manager.send_error(connection_id, f"Play command error: {str(e)}")

    async def _handle_stop_command(self, connection_id: str, data: dict):
        """Handle stop command"""
        try:
            from ...services.playback_service import get_playback_service

            playback_service = await get_playback_service()
            result = await playback_service.stop(connection_id=connection_id)

            await websocket_manager.send_message(connection_id, {
                "type": "control_response",
                "data": {
                    "command": "stop",
                    "success": result["success"],
                    "result": result
                }
            })

        except Exception as e:
            await websocket_manager.send_error(connection_id, f"Stop command error: {str(e)}")

    async def _handle_pause_command(self, connection_id: str, data: dict):
        """Handle pause command"""
        try:
            from ...services.playback_service import get_playback_service

            playback_service = await get_playback_service()
            result = await playback_service.pause(connection_id=connection_id)

            await websocket_manager.send_message(connection_id, {
                "type": "control_response",
                "data": {
                    "command": "pause",
                    "success": result["success"],
                    "result": result
                }
            })

        except Exception as e:
            await websocket_manager.send_error(connection_id, f"Pause command error: {str(e)}")

    async def _handle_resume_command(self, connection_id: str, data: dict):
        """Handle resume command"""
        try:
            from ...services.playback_service import get_playback_service

            playback_service = await get_playback_service()
            result = await playback_service.resume(connection_id=connection_id)

            await websocket_manager.send_message(connection_id, {
                "type": "control_response",
                "data": {
                    "command": "resume",
                    "success": result["success"],
                    "result": result
                }
            })

        except Exception as e:
            await websocket_manager.send_error(connection_id, f"Resume command error: {str(e)}")

    async def _handle_seek_command(self, connection_id: str, data: dict):
        """Handle seek command"""
        try:
            from ...services.playback_service import get_playback_service

            position = data.get("position", 0.0)
            playback_service = await get_playback_service()
            result = await playback_service.seek(position=position, connection_id=connection_id)

            await websocket_manager.send_message(connection_id, {
                "type": "control_response",
                "data": {
                    "command": "seek",
                    "success": result["success"],
                    "result": result
                }
            })

        except Exception as e:
            await websocket_manager.send_error(connection_id, f"Seek command error: {str(e)}")

    async def _handle_record_start_command(self, connection_id: str, data: dict):
        """Handle record start command"""
        try:
            from ...services.recording_service import get_recording_service
            import uuid
            from decimal import Decimal

            track_id = data.get("track_id")
            device_id = data.get("device_id")
            start_time = data.get("start_time", 0.0)

            if not track_id or not device_id:
                await websocket_manager.send_error(connection_id, "Missing track_id or device_id")
                return

            recording_service = await get_recording_service()
            result = await recording_service.start_recording(
                track_id=uuid.UUID(track_id),
                device_id=device_id,
                start_time=Decimal(str(start_time)),
                connection_id=connection_id
            )

            await websocket_manager.send_message(connection_id, {
                "type": "control_response",
                "data": {
                    "command": "record_start",
                    "success": result["success"],
                    "result": result
                }
            })

        except Exception as e:
            await websocket_manager.send_error(connection_id, f"Record start command error: {str(e)}")

    async def _handle_record_stop_command(self, connection_id: str, data: dict):
        """Handle record stop command"""
        try:
            from ...services.recording_service import get_recording_service
            import uuid

            session_id = data.get("session_id")

            if not session_id:
                await websocket_manager.send_error(connection_id, "Missing session_id")
                return

            recording_service = await get_recording_service()
            result = await recording_service.stop_and_save_recording(
                session_id=uuid.UUID(session_id),
                connection_id=connection_id
            )

            await websocket_manager.send_message(connection_id, {
                "type": "control_response",
                "data": {
                    "command": "record_stop",
                    "success": result["success"],
                    "result": result
                }
            })

        except Exception as e:
            await websocket_manager.send_error(connection_id, f"Record stop command error: {str(e)}")

    async def _handle_loop_enable_command(self, connection_id: str, data: dict):
        """Handle loop enable command"""
        try:
            from ...services.loop_service import get_loop_service
            import uuid

            loop_id = data.get("loop_id")

            if not loop_id:
                await websocket_manager.send_error(connection_id, "Missing loop_id")
                return

            loop_service = await get_loop_service()
            result = await loop_service.enable_loop(
                loop_id=uuid.UUID(loop_id),
                connection_id=connection_id
            )

            await websocket_manager.send_message(connection_id, {
                "type": "control_response",
                "data": {
                    "command": "loop_enable",
                    "success": result["success"],
                    "result": result
                }
            })

        except Exception as e:
            await websocket_manager.send_error(connection_id, f"Loop enable command error: {str(e)}")

    async def _handle_loop_disable_command(self, connection_id: str, data: dict):
        """Handle loop disable command"""
        try:
            from ...services.loop_service import get_loop_service

            loop_service = await get_loop_service()
            result = await loop_service.disable_loop(connection_id=connection_id)

            await websocket_manager.send_message(connection_id, {
                "type": "control_response",
                "data": {
                    "command": "loop_disable",
                    "success": result["success"],
                    "result": result
                }
            })

        except Exception as e:
            await websocket_manager.send_error(connection_id, f"Loop disable command error: {str(e)}")


# Global instances
websocket_manager = ConnectionManager()
audio_stream_handler = AudioStreamHandler()


async def initialize_websocket_callbacks():
    """Initialize WebSocket callbacks with services for real-time updates"""
    try:
        # Import services
        from ..services.playback_service import get_playback_service
        from ..services.recording_service import get_recording_service
        from ..services.loop_service import get_loop_service

        # Create master callback that routes to appropriate WebSocket methods
        async def master_websocket_callback(connection_id: str, message: dict):
            """Master callback for routing WebSocket messages"""
            message_type = message.get("type")

            if message_type == "position_update":
                # Broadcast position updates to project
                project_id = message["data"].get("project_id")
                position = message["data"].get("position")
                if project_id:
                    await websocket_manager.send_position_update(project_id, position)

            elif message_type in ["playback_started", "playback_stopped", "playback_paused",
                                 "playback_resumed", "playback_seeked"]:
                # Send to specific connection and project
                if connection_id:
                    await websocket_manager.send_message(connection_id, message)

            elif message_type in ["recording_started", "recording_stopped", "recording_progress",
                                 "recording_saved", "recording_error"]:
                # Send recording updates
                if connection_id:
                    await websocket_manager.send_recording_status_update(connection_id, message["data"])

            elif message_type in ["loop_enabled", "loop_disabled", "loop_started", "loop_ended",
                                 "loop_region_created", "loop_region_deleted"]:
                # Send loop events to project
                project_id = message["data"].get("project_id")
                if project_id:
                    await websocket_manager.send_loop_event(
                        project_id,
                        message_type,
                        message["data"]
                    )

            elif message_type in ["all_recordings_stopped"]:
                # Broadcast system notifications
                for project_id in websocket_manager.project_connections:
                    await websocket_manager.send_system_notification(
                        "All recordings stopped",
                        level="info",
                        project_id=project_id
                    )

            else:
                # Send generic message to specific connection
                if connection_id:
                    await websocket_manager.send_message(connection_id, message)

        # Initialize services with WebSocket callback
        try:
            playback_service = await get_playback_service()
            playback_service.set_websocket_callback(master_websocket_callback)
        except Exception as e:
            websocket_logger.log_warning(f"Failed to initialize playback WebSocket callback: {e}")

        try:
            recording_service = await get_recording_service()
            recording_service.set_websocket_callback(master_websocket_callback)
        except Exception as e:
            websocket_logger.log_warning(f"Failed to initialize recording WebSocket callback: {e}")

        try:
            loop_service = await get_loop_service()
            loop_service.set_websocket_callback(master_websocket_callback)
        except Exception as e:
            websocket_logger.log_warning(f"Failed to initialize loop WebSocket callback: {e}")

        websocket_logger.log_info("WebSocket callbacks initialized successfully")

    except Exception as e:
        websocket_logger.log_error(f"Failed to initialize WebSocket callbacks: {e}")


async def cleanup_websocket_connections():
    """Cleanup all WebSocket connections and resources"""
    try:
        # Disconnect all active connections
        active_connections = list(websocket_manager.active_connections.keys())
        for connection_id in active_connections:
            await websocket_manager.disconnect(connection_id)

        # Clear connection tracking
        websocket_manager.active_connections.clear()
        websocket_manager.project_connections.clear()

        # Clear audio streams
        audio_stream_handler.active_streams.clear()

        websocket_logger.log_info("WebSocket cleanup completed")

    except Exception as e:
        websocket_logger.log_error(f"Error during WebSocket cleanup: {e}")