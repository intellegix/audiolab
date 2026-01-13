# AudioLab Overdubbing & Looping Implementation Plan

## Overview
Transform AudioLab into a full-featured DAW with real-time recording, overdubbing, and looping capabilities for musicians to stack instruments and create layered compositions.

## Current State Analysis

### ✅ Foundation Ready
- **Database**: 32-track support, muting/soloing, timeline positioning
- **Audio Processing**: Professional 48kHz/24-bit pipeline, effects chains
- **WebSocket**: Real-time communication framework
- **Track Management**: Volume, pan, effects routing

### ❌ Missing Core Features
- Real-time audio input/recording
- Playback engine synchronization
- Overdubbing workflow
- Loop region management
- Input monitoring while recording

## Phase 1: Recording Engine Foundation (Week 1-2)

### 1.1 Audio Input System
**Files to Create:**
- `src/core/audio_input.py` - Audio input device management
- `src/core/recording_session.py` - Recording session lifecycle
- `src/services/recording_service.py` - High-level recording orchestration

**Database Extensions:**
```python
# Add to src/database/models.py
class RecordingSession(Base):
    __tablename__ = "recording_sessions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    track_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tracks.id"))
    input_device_id: Mapped[str] = mapped_column(String(255), nullable=False)
    start_time: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)
    duration: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="recording")  # recording, stopped, saved
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())

# Extend Track model
class Track(Base):
    # ... existing fields ...
    record_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    input_device_id: Mapped[str] = mapped_column(String(255), nullable=True)
    monitoring_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
```

**Core Audio Input Classes:**
```python
# src/core/audio_input.py
class AudioInputDevice:
    def __init__(self, device_id: str, name: str, sample_rate: int = 48000):
        self.device_id = device_id
        self.name = name
        self.sample_rate = sample_rate
        self.stream = None

    async def start_recording(self, callback: Callable):
        """Start real-time audio input stream"""

    async def stop_recording(self):
        """Stop audio input and cleanup"""

class AudioInputManager:
    async def enumerate_devices(self) -> List[AudioInputDevice]:
        """List available audio input devices"""

    async def get_device(self, device_id: str) -> AudioInputDevice:
        """Get specific input device"""
```

### 1.2 Playback Engine
**Files to Create:**
- `src/core/playback_engine.py` - Synchronized playback system
- `src/services/playback_service.py` - Playback orchestration

**Playback Engine Classes:**
```python
# src/core/playback_engine.py
class PlaybackEngine:
    def __init__(self):
        self.current_position: float = 0.0  # Seconds
        self.is_playing: bool = False
        self.tempo: Decimal = Decimal("120.0")
        self.tracks: Dict[uuid.UUID, Track] = {}
        self.position_callbacks: List[Callable] = []

    async def play(self, start_position: float = 0.0):
        """Start playback from position"""

    async def stop(self):
        """Stop playback"""

    async def seek_to(self, position: float):
        """Jump to specific time position"""

    async def get_current_position(self) -> float:
        """Get current playback position in seconds"""

    def add_position_callback(self, callback: Callable[[float], None]):
        """Add callback for position updates (WebSocket sync)"""
```

### 1.3 API Endpoints
**Extend src/api/routes/audio.py:**
```python
@router.get("/devices", response_model=List[AudioInputDeviceResponse])
async def get_audio_input_devices():
    """List available audio input devices for recording"""

@router.post("/tracks/{track_id}/record/start")
async def start_track_recording(
    track_id: uuid.UUID,
    device_id: str,
    connection_id: Optional[str] = Query(None)
):
    """Start recording on a track with specified input device"""

@router.post("/tracks/{track_id}/record/stop")
async def stop_track_recording(track_id: uuid.UUID):
    """Stop recording and save audio clip"""

@router.post("/projects/{project_id}/playback/play")
async def start_playback(project_id: uuid.UUID, position: float = 0.0):
    """Start project playback from position"""

@router.post("/projects/{project_id}/playback/stop")
async def stop_playback(project_id: uuid.UUID):
    """Stop project playback"""

@router.get("/projects/{project_id}/playback/position")
async def get_playback_position(project_id: uuid.UUID) -> float:
    """Get current playback position"""
```

## Phase 2: Overdubbing Workflow (Week 3)

### 2.1 Synchronized Recording + Playback
**Files to Enhance:**
- `src/core/sync_engine.py` - Keep recording and playback in sync
- `src/api/websocket.py` - Real-time position updates

**Synchronization Engine:**
```python
# src/core/sync_engine.py
class SyncEngine:
    def __init__(self, playback_engine: PlaybackEngine):
        self.playback_engine = playback_engine
        self.active_recordings: Dict[uuid.UUID, RecordingSession] = {}

    async def start_overdub(self, track_id: uuid.UUID, device_id: str):
        """Start recording on track while maintaining playback"""
        # 1. Start recording session
        # 2. Ensure playback is synchronized
        # 3. Mix playback audio for monitoring

    async def sync_position_updates(self):
        """Send position updates via WebSocket every 50ms"""
```

**WebSocket Real-time Updates:**
```python
# Enhance src/api/websocket.py
async def send_playback_position(self, project_id: str, position: float):
    """Broadcast playback position to all project clients"""

async def send_recording_status(self, track_id: str, status: str):
    """Notify clients of recording state changes"""

async def _handle_audio_data(self, connection_id: str, message: dict):
    """Process incoming audio data from client (if using browser recording)"""
    # Route audio chunks to appropriate recording session

async def _handle_control_message(self, connection_id: str, message: dict):
    """Handle play/stop/record commands"""
    if message["command"] == "play":
        await playback_service.play(message.get("position", 0.0))
    elif message["command"] == "stop":
        await playback_service.stop()
    elif message["command"] == "record":
        await recording_service.start_recording(
            message["track_id"],
            message["device_id"]
        )
```

### 2.2 Input Monitoring
**Files to Create:**
- `src/core/monitoring.py` - Direct input monitoring system

**Monitoring System:**
```python
# src/core/monitoring.py
class InputMonitor:
    def __init__(self):
        self.monitoring_enabled = True
        self.monitor_level = 0.7  # Monitor mix level

    async def enable_direct_monitoring(self, input_device: AudioInputDevice):
        """Pass input directly to output for low-latency monitoring"""

    async def mix_with_playback(self, input_audio: np.ndarray, playback_audio: np.ndarray):
        """Mix input and playback for headphone monitoring"""
```

## Phase 3: Looping System (Week 4)

### 3.1 Loop Region Model
**Database Extension:**
```python
# Add to src/database/models.py
class LoopRegion(Base):
    __tablename__ = "loop_regions"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    project_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(String(255), nullable=False, default="Loop Region")
    start_time: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)
    end_time: Mapped[Decimal] = mapped_column(DECIMAL(10, 6), nullable=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    repeat_count: Mapped[int] = mapped_column(Integer, nullable=True)  # None = infinite
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, server_default=func.now())
```

### 3.2 Loop Playback Logic
**Files to Enhance:**
- `src/core/playback_engine.py` - Add loop boundary detection
- `src/core/loop_manager.py` - Loop region management

**Loop Management:**
```python
# src/core/loop_manager.py
class LoopManager:
    def __init__(self, playback_engine: PlaybackEngine):
        self.playback_engine = playback_engine
        self.active_loop: Optional[LoopRegion] = None
        self.loop_count = 0

    async def enable_loop(self, loop_region: LoopRegion):
        """Enable looping for specified region"""

    async def check_loop_boundary(self, current_position: float):
        """Check if playback hit loop end, restart if needed"""
        if self.active_loop and current_position >= float(self.active_loop.end_time):
            await self.playback_engine.seek_to(float(self.active_loop.start_time))
            self.loop_count += 1
```

### 3.3 Punch Recording
**Files to Create:**
- `src/core/punch_recording.py` - Automatic punch-in/out recording

**Punch Recording System:**
```python
# src/core/punch_recording.py
class PunchRecorder:
    def __init__(self, sync_engine: SyncEngine):
        self.sync_engine = sync_engine
        self.punch_regions: List[PunchRegion] = []

    async def setup_punch_recording(self, track_id: uuid.UUID,
                                  punch_in: float, punch_out: float):
        """Setup automatic recording in specified time region"""

    async def check_punch_boundaries(self, current_position: float):
        """Auto-start/stop recording based on position"""
```

## Phase 4: Professional Features (Week 5)

### 4.1 Quantization & Grid Snap
```python
# src/core/quantization.py
class QuantizeEngine:
    def __init__(self, tempo: Decimal, time_signature: str = "4/4"):
        self.tempo = tempo
        self.time_signature = time_signature

    def snap_to_grid(self, position: float, grid_division: str = "1/16") -> float:
        """Snap timeline position to beat grid"""

    def quantize_clip_start(self, clip: Clip, grid_division: str = "1/16"):
        """Quantize clip start time to beat grid"""
```

### 4.2 Input Level Metering
```python
# src/core/metering.py
class InputMeter:
    def __init__(self):
        self.peak_level = 0.0
        self.rms_level = 0.0
        self.clip_detected = False

    async def analyze_input(self, audio_data: np.ndarray):
        """Real-time level analysis"""

    async def send_meter_data_via_websocket(self, connection_id: str):
        """Send meter levels to client for visual feedback"""
```

## Implementation Priority

### **Week 1: Core Recording**
1. Audio input device enumeration
2. Basic recording session management
3. Audio file saving after recording

### **Week 2: Playback Engine**
1. Multi-track playback mixing
2. Position tracking and synchronization
3. WebSocket position broadcasts

### **Week 3: Overdubbing**
1. Simultaneous record + playback
2. Input monitoring while recording
3. Overdubbing workflow API

### **Week 4: Looping**
1. Loop region database model
2. Loop boundary detection and restart
3. Punch-in/punch-out recording

### **Week 5: Polish**
1. Quantization and beat grid
2. Input level metering
3. Performance optimization

## Technology Stack

### **Already Available:**
- ✅ **PyAudio** - Real-time audio I/O (in requirements.txt)
- ✅ **NumPy** - Audio buffer processing
- ✅ **WebSocket** - Real-time communication
- ✅ **PostgreSQL** - Metadata storage
- ✅ **FastAPI** - API endpoints

### **Additional Dependencies:**
```python
# Add to requirements.txt
rtaudio==1.4.5           # Alternative to PyAudio with better latency
sounddevice==0.4.6       # Modern Python audio I/O library
mido==1.2.10            # MIDI support for tempo sync
```

## Expected User Workflow

### **Recording Lead Guitar Over Rhythm:**
1. **Load Project** with existing rhythm guitar track
2. **Create New Track** for lead guitar
3. **Select Input Device** (audio interface/microphone)
4. **Enable Loop Region** for 8-bar section
5. **Start Playback + Recording** - rhythm plays while recording lead
6. **Monitor Input** through headphones mixed with playback
7. **Auto-Loop** at end of 8 bars, keep recording layers
8. **Stop When Satisfied** - lead guitar saved as new clip
9. **Adjust Levels** and mix lead with rhythm
10. **Export Final Mix** with both guitars

### **Stack Multiple Instruments:**
1. Record drums on Track 1
2. Overdub bass on Track 2 while drums play
3. Overdub rhythm guitar on Track 3 while drums+bass play
4. Overdub lead guitar on Track 4 while full backing plays
5. Overdub vocals on Track 5 while full band plays
6. Mix all tracks and export final song

## Testing Strategy

### **Unit Tests:**
- Audio input device enumeration
- Recording session lifecycle
- Playback position accuracy
- Loop boundary detection

### **Integration Tests:**
- Record + playback synchronization
- WebSocket real-time updates
- Multi-track overdubbing workflow
- Loop recording with multiple takes

### **Performance Tests:**
- Audio latency measurement (target <10ms)
- Memory usage during 32-track playback
- CPU usage during overdubbing
- Buffer underrun/overrun handling

## Success Metrics

### **Functional Requirements:**
- ✅ Record audio while playing existing tracks
- ✅ Maintain <10ms latency for monitoring
- ✅ Support 32 simultaneous tracks
- ✅ Loop recording with automatic punch-in/out
- ✅ Real-time position sync across multiple clients
- ✅ Input level metering and clip detection

### **Professional Standards:**
- ✅ 48kHz/24-bit audio quality maintained
- ✅ Sample-accurate timing for overdubs
- ✅ Non-destructive recording (creates new clips)
- ✅ Beat-grid quantization for timing correction
- ✅ Direct monitoring for zero-latency input feedback

This implementation transforms AudioLab into a professional DAW capable of multi-track overdubbing, looping, and collaborative music production workflows.