# AudioLab - Professional Audio Production Suite

**Complete audio production suite with AI stem separation, real-time mixing, and cross-platform desktop deployment.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/typescript-5.2+-blue.svg)](https://www.typescriptlang.org/)
[![Tauri 2.0](https://img.shields.io/badge/tauri-2.0+-orange.svg)](https://tauri.app/)

## Features

- 🎵 **AI-Powered Stem Separation** - Demucs v4 integration for vocals, drums, bass, guitar, piano separation
- 🎛️ **Professional DAW Interface** - Multi-track mixing with real-time effects processing
- 🖥️ **Cross-Platform Desktop** - Native Windows, macOS, and Linux support via Tauri
- ⚡ **Real-Time Processing** - Sub-10ms latency for professional monitoring
- 🎚️ **Advanced Effects** - Parametric EQ, compression, reverb, and mastering tools
- 📊 **Project Management** - Complete session management with automatic saving
- 🔊 **Export & Mastering** - Multi-format export with loudness targeting (-14 LUFS)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   AudioLab Desktop App                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   React UI      │  │   Tauri 2.0     │  │  Audio I/O  │ │
│  │   (Frontend)    │  │   (Desktop)     │  │   Engine    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │ WebSocket/HTTP
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Backend Server                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   HTTP API  │  │  WebSocket  │  │   Audio Processing  │ │
│  │   Endpoints │  │   Handler   │  │      Pipeline       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data & Storage Layer                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ PostgreSQL  │  │    Redis    │  │   File Storage      │ │
│  │ (Metadata)  │  │   (Cache)   │  │   (Audio Files)     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                AI Processing Services                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Demucs v4   │  │  Audio FX   │  │    Mastering       │ │
│  │ Separation  │  │ Processing  │  │     Engine         │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Backend (Python 70%)
- **FastAPI** - Async web framework with WebSocket support
- **Demucs v4** - Meta's AI stem separation (9.20 dB SDR quality)
- **Librosa** - Advanced audio analysis and processing
- **PyDub** - Audio manipulation and format conversion
- **SoundFile** - High-performance audio I/O
- **PostgreSQL 17** - Project and metadata storage
- **Redis** - Real-time session management

### Frontend (TypeScript 25%)
- **React 18** - Component-based UI framework
- **Tauri 2.0** - Cross-platform desktop (10MB vs 100MB+ Electron)
- **TailwindCSS** - Utility-first CSS framework
- **Zustand** - Client state management
- **React Query** - Server state and caching
- **Web Audio API** - Real-time audio visualization

## Quick Start

### Prerequisites

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **Rust 1.70+** (for Tauri)
- **Docker & Docker Compose** (for databases)
- **Git**

### 1. Clone Repository

```bash
git clone https://github.com/intellegix/audiolab.git
cd audiolab
```

### 2. Set Up Backend

```bash
# Create virtual environment
python -m venv audiolab-env

# Activate virtual environment
# Windows:
audiolab-env\\Scripts\\activate
# macOS/Linux:
source audiolab-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Demucs v4
pip install -U demucs
```

### 3. Set Up Database Services

```bash
# Start PostgreSQL and Redis
docker-compose up -d postgresql redis

# Verify services are running
docker-compose ps
```

### 4. Set Up Frontend

```bash
# Install frontend dependencies
npm install

# Install Tauri CLI
npm install -g @tauri-apps/cli
```

### 5. Run Development Environment

```bash
# Terminal 1: Start FastAPI backend
cd src
python -m uvicorn main:app --reload --port 8000

# Terminal 2: Start Tauri frontend
npm run tauri:dev
```

The AudioLab desktop application should open automatically.

## Project Structure

```
AudioLab/
├── .claude/                     # Master plan and patterns
│   ├── CLAUDE.md               # Master project instructions
│   └── patterns/               # Reference implementations
├── src/                        # Python backend
│   ├── core/                  # Audio processing engine
│   ├── api/                   # FastAPI endpoints
│   ├── database/              # PostgreSQL models
│   ├── services/              # Business logic
│   └── utils/                 # Shared utilities
├── frontend/                   # React + Tauri frontend
│   ├── src/
│   │   ├── components/        # React components
│   │   ├── store/             # Zustand state management
│   │   ├── hooks/             # Custom React hooks
│   │   └── utils/             # Frontend utilities
│   └── src-tauri/             # Tauri Rust backend
├── docker/                    # Database configurations
├── tests/                     # Test suites
└── docs/                      # Documentation
```

## Usage

### 1. Create New Project

1. Launch AudioLab
2. Click "New Project"
3. Configure sample rate (48kHz recommended)
4. Set project name and location

### 2. Import Audio

1. Drag audio files into the timeline
2. Or use File → Import Audio
3. Supported formats: WAV, FLAC, MP3, AIFF

### 3. AI Stem Separation

1. Right-click audio clip
2. Select "Separate Stems"
3. Choose model:
   - `htdemucs_ft` - 4 stems (vocals, drums, bass, other)
   - `htdemucs_6s` - 6 stems (adds guitar and piano)
4. Processing takes ~1.5x real-time with GPU

### 4. Mixing

1. Adjust track levels and panning
2. Add effects from the effects panel
3. Use bus sends for reverb and delay
4. Real-time monitoring with <10ms latency

### 5. Export

1. File → Export → Mix
2. Choose format (WAV, FLAC, MP3)
3. Set loudness target (-14 LUFS for streaming)
4. Export stems or master mix

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Database
DATABASE_URL=postgresql://audiolab_user:audiolab_password@localhost:5432/audiolab
REDIS_URL=redis://localhost:6379

# Audio Processing
DEFAULT_SAMPLE_RATE=48000
DEFAULT_BUFFER_SIZE=512
MAX_TRACKS=32

# AI Models
DEMUCS_MODEL_PATH=./models/
DEMUCS_DEFAULT_MODEL=htdemucs_ft

# Development
LOG_LEVEL=INFO
DEBUG=false
```

### Audio Settings

Optimize for your system in Settings → Audio:

- **Buffer Size**: 256-512 samples (lower = less latency, higher CPU)
- **Sample Rate**: 48kHz (professional standard)
- **Device**: Choose your audio interface

## Development

### Adding New Effects

1. Create effect class in `src/core/effects/`
2. Inherit from `BaseAudioProcessor`
3. Implement `_process_internal()` method
4. Add to effects registry
5. Create UI component in `frontend/src/components/effects/`

Example:

```python
class CustomReverb(BaseAudioProcessor):
    async def _process_internal(self, audio: np.ndarray, **kwargs) -> ProcessingResult:
        # Your effect processing here
        return ProcessingResult(success=True, data=processed_audio)
```

### Testing

```bash
# Backend tests
pytest tests/

# Frontend tests
npm run test

# Audio processing tests
pytest tests/audio/ -v
```

### Building for Production

```bash
# Build cross-platform releases
npm run build:all-platforms

# Individual platforms
npm run tauri:build:windows
npm run tauri:build:macos
npm run tauri:build:linux
```

## Performance

### Audio Quality Benchmarks
- **Latency**: <10ms end-to-end monitoring
- **CPU Usage**: <30% for 8-track projects with effects
- **Memory Usage**: <200MB for typical projects
- **Stem Separation**: >8.5 SDR for guitar, >9.0 SDR for vocals

### System Requirements

**Minimum:**
- 4GB RAM
- Dual-core CPU
- 500MB storage
- Audio interface (recommended)

**Recommended:**
- 8GB+ RAM
- Quad-core CPU
- NVIDIA GTX 1060+ (for AI processing)
- 2GB+ storage
- Professional audio interface

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/audiolab-new-feature`
3. Follow coding standards in `.claude/patterns/`
4. Run tests: `pytest && npm run test`
5. Submit pull request

### Coding Standards

- **Python**: Type hints on all functions, async/await for I/O
- **TypeScript**: Explicit return types, strict mode enabled
- **Commits**: Conventional commits (`feat:`, `fix:`, `refactor:`)
- **Documentation**: Update pattern files for new features

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/intellegix/audiolab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/intellegix/audiolab/discussions)

## Roadmap

### Phase 1 (Current)
- ✅ Core audio engine and Demucs integration
- ✅ Basic DAW interface with multi-track mixing
- ✅ Cross-platform desktop deployment
- 🔄 Real-time effects processing

### Phase 2 (Q2 2024)
- 🔲 VST plugin support
- 🔲 MIDI integration
- 🔲 Advanced automation
- 🔲 Cloud project sync

### Phase 3 (Q3 2024)
- 🔲 Real-time collaboration
- 🔲 Plugin marketplace
- 🔲 Advanced AI features
- 🔲 Mobile companion app

---

**AudioLab** - Professional audio production, powered by AI.

*Built with ❤️ by Austin Kidwell | Intellegix, ASR Inc*