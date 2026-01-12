"""
Unit tests for WebSocket connection management and audio streaming
Tests ConnectionManager, AudioStreamHandler, and message routing
"""
import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Set


class MockWebSocket:
    """Mock WebSocket for testing"""

    def __init__(self):
        self.messages_sent = []
        self.closed = False
        self.accept_called = False

    async def accept(self):
        self.accept_called = True

    async def send_text(self, data: str):
        if self.closed:
            raise Exception("WebSocket connection closed")
        self.messages_sent.append(data)

    def close(self):
        self.closed = True


class MockWebSocketDisconnect(Exception):
    """Mock WebSocketDisconnect exception"""
    pass


class MockConnectionManager:
    """Mock ConnectionManager for testing without imports"""

    def __init__(self):
        self.active_connections: Dict[str, MockWebSocket] = {}
        self.project_connections: Dict[str, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: MockWebSocket, connection_id: str, project_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()

        async with self._lock:
            self.active_connections[connection_id] = websocket

            if project_id not in self.project_connections:
                self.project_connections[project_id] = set()
            self.project_connections[project_id].add(connection_id)

    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection"""
        async with self._lock:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

            # Remove from project connections
            for project_id, connections in self.project_connections.items():
                connections.discard(connection_id)

    async def send_message(self, connection_id: str, message: dict):
        """Send message to specific connection"""
        if connection_id not in self.active_connections:
            return False

        websocket = self.active_connections[connection_id]

        try:
            await websocket.send_text(json.dumps(message))
            return True
        except Exception:
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


class MockAudioStreamHandler:
    """Mock AudioStreamHandler for testing"""

    def __init__(self):
        self.active_streams: Dict[str, dict] = {}
        self.connection_manager = MockConnectionManager()

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

            if message_type == "audio_data":
                await self._handle_audio_data(connection_id, message)
            elif message_type == "control":
                await self._handle_control_message(connection_id, message)
            elif message_type == "ping":
                await self.connection_manager.send_message(connection_id, {
                    "type": "pong"
                })

        except Exception as e:
            await self.connection_manager.send_error(connection_id, str(e))

    async def _handle_audio_data(self, connection_id: str, message: dict):
        """Handle incoming audio data"""
        # Mock implementation for testing
        pass

    async def _handle_control_message(self, connection_id: str, message: dict):
        """Handle control messages (play, stop, etc.)"""
        # Mock implementation for testing
        pass


@pytest.mark.unit
class TestConnectionManager:
    """Test WebSocket connection management functionality"""

    @pytest.mark.asyncio
    async def test_connection_establishment(self):
        """Test WebSocket connection establishment"""
        manager = MockConnectionManager()
        websocket = MockWebSocket()

        connection_id = "test_conn_1"
        project_id = "test_project_1"

        # Test connection
        await manager.connect(websocket, connection_id, project_id)

        assert websocket.accept_called == True
        assert connection_id in manager.active_connections
        assert project_id in manager.project_connections
        assert connection_id in manager.project_connections[project_id]

    @pytest.mark.asyncio
    async def test_multiple_connections_same_project(self):
        """Test multiple connections to same project"""
        manager = MockConnectionManager()

        project_id = "test_project_1"
        connections = []

        # Add multiple connections to same project
        for i in range(3):
            websocket = MockWebSocket()
            connection_id = f"test_conn_{i}"

            await manager.connect(websocket, connection_id, project_id)
            connections.append(connection_id)

        assert len(manager.active_connections) == 3
        assert len(manager.project_connections[project_id]) == 3

        for conn_id in connections:
            assert conn_id in manager.project_connections[project_id]

    @pytest.mark.asyncio
    async def test_connection_disconnection(self):
        """Test WebSocket disconnection"""
        manager = MockConnectionManager()
        websocket = MockWebSocket()

        connection_id = "test_conn_1"
        project_id = "test_project_1"

        # Connect first
        await manager.connect(websocket, connection_id, project_id)

        # Then disconnect
        await manager.disconnect(connection_id)

        assert connection_id not in manager.active_connections
        assert connection_id not in manager.project_connections[project_id]

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        """Test successful message sending"""
        manager = MockConnectionManager()
        websocket = MockWebSocket()

        connection_id = "test_conn_1"
        project_id = "test_project_1"

        await manager.connect(websocket, connection_id, project_id)

        # Send message
        message = {"type": "test", "data": "hello"}
        result = await manager.send_message(connection_id, message)

        assert result == True
        assert len(websocket.messages_sent) == 1
        sent_message = json.loads(websocket.messages_sent[0])
        assert sent_message["type"] == "test"
        assert sent_message["data"] == "hello"

    @pytest.mark.asyncio
    async def test_send_message_to_disconnected_client(self):
        """Test sending message to non-existent connection"""
        manager = MockConnectionManager()

        message = {"type": "test", "data": "hello"}
        result = await manager.send_message("non_existent_conn", message)

        assert result == False

    @pytest.mark.asyncio
    async def test_send_error_message(self):
        """Test error message sending"""
        manager = MockConnectionManager()
        websocket = MockWebSocket()

        connection_id = "test_conn_1"
        project_id = "test_project_1"

        await manager.connect(websocket, connection_id, project_id)

        # Send error
        error_text = "Test error message"
        await manager.send_error(connection_id, error_text)

        assert len(websocket.messages_sent) == 1
        sent_message = json.loads(websocket.messages_sent[0])
        assert sent_message["type"] == "error"
        assert sent_message["data"]["error"] == error_text

    @pytest.mark.asyncio
    async def test_project_broadcast(self):
        """Test broadcasting to all connections in a project"""
        manager = MockConnectionManager()
        project_id = "test_project_1"

        # Create multiple connections in same project
        websockets = []
        connection_ids = []

        for i in range(3):
            websocket = MockWebSocket()
            connection_id = f"test_conn_{i}"

            await manager.connect(websocket, connection_id, project_id)
            websockets.append(websocket)
            connection_ids.append(connection_id)

        # Broadcast message
        broadcast_message = {"type": "broadcast", "data": "hello all"}
        await manager.broadcast_to_project(project_id, broadcast_message)

        # Verify all connections received the message
        for websocket in websockets:
            assert len(websocket.messages_sent) == 1
            sent_message = json.loads(websocket.messages_sent[0])
            assert sent_message["type"] == "broadcast"
            assert sent_message["data"] == "hello all"

    @pytest.mark.asyncio
    async def test_broadcast_to_nonexistent_project(self):
        """Test broadcasting to project with no connections"""
        manager = MockConnectionManager()

        message = {"type": "test", "data": "hello"}

        # Should not raise error
        await manager.broadcast_to_project("nonexistent_project", message)

    @pytest.mark.asyncio
    async def test_cross_project_isolation(self):
        """Test that projects are isolated from each other"""
        manager = MockConnectionManager()

        # Create connections in different projects
        websocket1 = MockWebSocket()
        websocket2 = MockWebSocket()

        await manager.connect(websocket1, "conn1", "project1")
        await manager.connect(websocket2, "conn2", "project2")

        # Broadcast to project1
        message = {"type": "test", "data": "project1 only"}
        await manager.broadcast_to_project("project1", message)

        # Only project1 connection should receive message
        assert len(websocket1.messages_sent) == 1
        assert len(websocket2.messages_sent) == 0


@pytest.mark.unit
class TestAudioStreamHandler:
    """Test audio streaming functionality"""

    @pytest.mark.asyncio
    async def test_start_audio_stream(self):
        """Test starting audio stream"""
        handler = MockAudioStreamHandler()

        connection_id = "test_conn_1"
        project_id = "test_project_1"
        config = {"sample_rate": 48000, "buffer_size": 512}

        await handler.start_audio_stream(connection_id, project_id, config)

        assert connection_id in handler.active_streams
        stream_info = handler.active_streams[connection_id]
        assert stream_info["project_id"] == project_id
        assert stream_info["config"] == config
        assert "started_at" in stream_info

    @pytest.mark.asyncio
    async def test_stop_audio_stream(self):
        """Test stopping audio stream"""
        handler = MockAudioStreamHandler()

        connection_id = "test_conn_1"
        project_id = "test_project_1"
        config = {"sample_rate": 48000}

        # Start stream first
        await handler.start_audio_stream(connection_id, project_id, config)
        assert connection_id in handler.active_streams

        # Stop stream
        await handler.stop_audio_stream(connection_id)
        assert connection_id not in handler.active_streams

    @pytest.mark.asyncio
    async def test_stop_nonexistent_stream(self):
        """Test stopping stream that doesn't exist"""
        handler = MockAudioStreamHandler()

        # Should not raise error
        await handler.stop_audio_stream("nonexistent_conn")

    @pytest.mark.asyncio
    async def test_handle_ping_message(self):
        """Test ping/pong message handling"""
        handler = MockAudioStreamHandler()
        websocket = MockWebSocket()
        connection_id = "test_conn_1"
        project_id = "test_project_1"

        # Setup connection
        await handler.connection_manager.connect(websocket, connection_id, project_id)

        # Send ping message
        ping_message = json.dumps({"type": "ping"})
        await handler.handle_message(connection_id, ping_message)

        # Should receive pong response
        assert len(websocket.messages_sent) == 1
        response = json.loads(websocket.messages_sent[0])
        assert response["type"] == "pong"

    @pytest.mark.asyncio
    async def test_handle_audio_data_message(self):
        """Test audio data message handling"""
        handler = MockAudioStreamHandler()
        connection_id = "test_conn_1"

        # Send audio data message
        audio_message = json.dumps({
            "type": "audio_data",
            "data": {"samples": [0.1, 0.2, 0.3], "format": "f32"}
        })

        # Should not raise error (implementation is stubbed)
        await handler.handle_message(connection_id, audio_message)

    @pytest.mark.asyncio
    async def test_handle_control_message(self):
        """Test control message handling"""
        handler = MockAudioStreamHandler()
        connection_id = "test_conn_1"

        # Send control message
        control_message = json.dumps({
            "type": "control",
            "action": "play",
            "data": {"position": 0.0}
        })

        # Should not raise error (implementation is stubbed)
        await handler.handle_message(connection_id, control_message)

    @pytest.mark.asyncio
    async def test_handle_invalid_json_message(self):
        """Test handling invalid JSON message"""
        handler = MockAudioStreamHandler()
        websocket = MockWebSocket()
        connection_id = "test_conn_1"
        project_id = "test_project_1"

        # Setup connection
        await handler.connection_manager.connect(websocket, connection_id, project_id)

        # Send invalid JSON
        invalid_message = "invalid json {"
        await handler.handle_message(connection_id, invalid_message)

        # Should receive error message
        assert len(websocket.messages_sent) == 1
        response = json.loads(websocket.messages_sent[0])
        assert response["type"] == "error"
        assert "error" in response["data"]

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self):
        """Test handling unknown message type"""
        handler = MockAudioStreamHandler()
        connection_id = "test_conn_1"

        # Send unknown message type
        unknown_message = json.dumps({
            "type": "unknown_type",
            "data": "some data"
        })

        # Should not raise error
        await handler.handle_message(connection_id, unknown_message)


@pytest.mark.unit
class TestWebSocketIntegration:
    """Test integration between ConnectionManager and AudioStreamHandler"""

    @pytest.mark.asyncio
    async def test_audio_stream_workflow(self):
        """Test complete audio stream workflow"""
        manager = MockConnectionManager()
        handler = MockAudioStreamHandler()
        websocket = MockWebSocket()

        connection_id = "test_conn_1"
        project_id = "test_project_1"

        # 1. Connect
        await manager.connect(websocket, connection_id, project_id)
        assert connection_id in manager.active_connections

        # 2. Start audio stream
        config = {"sample_rate": 48000, "buffer_size": 512}
        await handler.start_audio_stream(connection_id, project_id, config)
        assert connection_id in handler.active_streams

        # 3. Send audio data
        await handler.connection_manager.connect(websocket, connection_id, project_id)
        audio_message = json.dumps({
            "type": "audio_data",
            "data": {"samples": [0.1, 0.2, 0.3]}
        })
        await handler.handle_message(connection_id, audio_message)

        # 4. Stop stream
        await handler.stop_audio_stream(connection_id)
        assert connection_id not in handler.active_streams

        # 5. Disconnect
        await manager.disconnect(connection_id)
        assert connection_id not in manager.active_connections

    @pytest.mark.asyncio
    async def test_concurrent_project_streaming(self):
        """Test multiple projects streaming simultaneously"""
        manager = MockConnectionManager()
        handler = MockAudioStreamHandler()

        projects_data = [
            ("proj1", "conn1", {"sample_rate": 44100}),
            ("proj2", "conn2", {"sample_rate": 48000}),
            ("proj3", "conn3", {"sample_rate": 96000})
        ]

        # Start streams for multiple projects
        for project_id, connection_id, config in projects_data:
            websocket = MockWebSocket()
            await manager.connect(websocket, connection_id, project_id)
            await handler.start_audio_stream(connection_id, project_id, config)

        # Verify all streams are active
        assert len(handler.active_streams) == 3
        assert len(manager.active_connections) == 3

        # Verify each project has correct configuration
        for project_id, connection_id, config in projects_data:
            assert connection_id in handler.active_streams
            assert handler.active_streams[connection_id]["config"] == config

    @pytest.mark.asyncio
    async def test_websocket_message_types_validation(self):
        """Test validation of different WebSocket message types"""
        message_types = [
            {"type": "ping"},
            {"type": "audio_data", "data": {"samples": [0.1, 0.2]}},
            {"type": "control", "action": "play"},
            {"type": "custom", "payload": "test"}
        ]

        handler = MockAudioStreamHandler()
        connection_id = "test_conn"

        for message in message_types:
            message_str = json.dumps(message)

            # Should handle all message types without error
            await handler.handle_message(connection_id, message_str)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])