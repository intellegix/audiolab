"""
Beat Generation Dependency Validation
Validate that all required dependencies for beat generation are available
"""

import importlib.util
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..core.logging import audio_logger
from ..core.result import Result


class BeatGenerationValidator:
    """Validator for beat generation dependencies and configuration"""

    REQUIRED_PACKAGES = {
        "transformers": "4.35.0",
        "torch": "1.11.0",
        "torchaudio": "0.11.0",
        "httpx": "0.25.0",
        "pretty_midi": "0.2.10"
    }

    OPTIONAL_PACKAGES = {
        "accelerate": "0.24.0",
        "xformers": "0.0.22",
        "mido": "1.3.0",
        "music21": "9.1.0"
    }

    @classmethod
    def validate_dependencies(cls) -> Result[Dict[str, bool]]:
        """Validate all beat generation dependencies"""

        validation_results = {
            "required": {},
            "optional": {},
            "providers": {},
            "gpu_support": False,
            "summary": {
                "all_required_available": True,
                "musicgen_ready": False,
                "soundraw_ready": False,
                "errors": []
            }
        }

        try:
            # Check required packages
            for package, min_version in cls.REQUIRED_PACKAGES.items():
                is_available, version, error = cls._check_package(package, min_version)
                validation_results["required"][package] = {
                    "available": is_available,
                    "version": version,
                    "min_version": min_version,
                    "error": error
                }

                if not is_available:
                    validation_results["summary"]["all_required_available"] = False
                    validation_results["summary"]["errors"].append(
                        f"Required package {package} not available: {error}"
                    )

            # Check optional packages
            for package, min_version in cls.OPTIONAL_PACKAGES.items():
                is_available, version, error = cls._check_package(package, min_version)
                validation_results["optional"][package] = {
                    "available": is_available,
                    "version": version,
                    "min_version": min_version,
                    "error": error
                }

            # Check provider-specific requirements
            validation_results["providers"]["musicgen"] = cls._validate_musicgen()
            validation_results["providers"]["soundraw"] = cls._validate_soundraw()

            # Check GPU support
            validation_results["gpu_support"] = cls._check_gpu_support()

            # Set readiness flags
            validation_results["summary"]["musicgen_ready"] = (
                validation_results["summary"]["all_required_available"] and
                validation_results["providers"]["musicgen"]["ready"]
            )

            validation_results["summary"]["soundraw_ready"] = (
                validation_results["required"]["httpx"]["available"] and
                validation_results["providers"]["soundraw"]["ready"]
            )

            audio_logger.log_processing_complete(
                operation="BeatGenerationValidation",
                duration_ms=0,
                musicgen_ready=validation_results["summary"]["musicgen_ready"],
                soundraw_ready=validation_results["summary"]["soundraw_ready"],
                gpu_support=validation_results["gpu_support"]
            )

            return Result.ok(validation_results)

        except Exception as e:
            audio_logger.log_processing_error(
                operation="BeatGenerationValidation",
                error=str(e)
            )
            return Result.err(f"Validation failed: {e}")

    @classmethod
    def _check_package(cls, package_name: str, min_version: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if a package is available and meets minimum version"""

        try:
            # Try to import the package
            spec = importlib.util.find_spec(package_name)
            if spec is None:
                return False, None, f"Package {package_name} not found"

            # Import and get version
            module = importlib.import_module(package_name)

            version = None
            for attr_name in ["__version__", "VERSION", "version"]:
                if hasattr(module, attr_name):
                    version = str(getattr(module, attr_name))
                    break

            if version is None:
                return True, "unknown", "Version could not be determined"

            # Simple version comparison (would use packaging.version in production)
            if cls._compare_versions(version, min_version) >= 0:
                return True, version, None
            else:
                return False, version, f"Version {version} is below minimum {min_version}"

        except ImportError as e:
            return False, None, f"Import error: {e}"
        except Exception as e:
            return False, None, f"Unexpected error: {e}"

    @classmethod
    def _compare_versions(cls, version1: str, version2: str) -> int:
        """Simple version comparison (-1: v1 < v2, 0: v1 == v2, 1: v1 > v2)"""

        try:
            # Extract major.minor.patch numbers
            def parse_version(v):
                parts = v.split('.')
                return [int(x) for x in parts[:3]]  # Only compare major.minor.patch

            v1_parts = parse_version(version1)
            v2_parts = parse_version(version2)

            # Pad to same length
            while len(v1_parts) < len(v2_parts):
                v1_parts.append(0)
            while len(v2_parts) < len(v1_parts):
                v2_parts.append(0)

            # Compare
            for v1, v2 in zip(v1_parts, v2_parts):
                if v1 < v2:
                    return -1
                elif v1 > v2:
                    return 1

            return 0

        except:
            # Fallback to string comparison
            return 0 if version1 == version2 else -1

    @classmethod
    def _validate_musicgen(cls) -> Dict[str, any]:
        """Validate MusicGen-specific requirements"""

        result = {
            "ready": False,
            "transformers_available": False,
            "torch_available": False,
            "model_loading_possible": False,
            "errors": []
        }

        try:
            # Check transformers with MusicGen support
            from transformers import MusicgenForConditionalGeneration, AutoProcessor
            result["transformers_available"] = True

            # Check torch/torchaudio
            import torch
            import torchaudio
            result["torch_available"] = True

            # Test basic model loading capability (without actually loading)
            try:
                # Just check that the class can be instantiated
                MusicgenForConditionalGeneration.from_pretrained.__doc__
                result["model_loading_possible"] = True
            except Exception as e:
                result["errors"].append(f"Model loading test failed: {e}")

            result["ready"] = (
                result["transformers_available"] and
                result["torch_available"] and
                result["model_loading_possible"]
            )

        except ImportError as e:
            result["errors"].append(f"MusicGen import failed: {e}")
        except Exception as e:
            result["errors"].append(f"MusicGen validation failed: {e}")

        return result

    @classmethod
    def _validate_soundraw(cls) -> Dict[str, any]:
        """Validate SOUNDRAW API requirements"""

        result = {
            "ready": False,
            "httpx_available": False,
            "api_key_configured": False,
            "errors": []
        }

        try:
            # Check httpx
            import httpx
            result["httpx_available"] = True

            # Check API key configuration (from environment or settings)
            from ..core.config import get_settings
            settings = get_settings()
            result["api_key_configured"] = settings.SOUNDRAW_API_KEY is not None

            if not result["api_key_configured"]:
                result["errors"].append("SOUNDRAW_API_KEY not configured")

            result["ready"] = (
                result["httpx_available"] and
                result["api_key_configured"]
            )

        except ImportError as e:
            result["errors"].append(f"SOUNDRAW dependencies import failed: {e}")
        except Exception as e:
            result["errors"].append(f"SOUNDRAW validation failed: {e}")

        return result

    @classmethod
    def _check_gpu_support(cls) -> bool:
        """Check if GPU support is available for beat generation"""

        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False

    @classmethod
    def validate_model_path(cls, provider: str, model_path: str) -> Result[bool]:
        """Validate model path exists and is writable"""

        try:
            path = Path(model_path)

            # Check if path exists or can be created
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    return Result.err(f"Cannot create model path {model_path}: {e}")

            # Check if writable
            if not path.is_dir():
                return Result.err(f"Model path {model_path} is not a directory")

            # Test write permissions
            test_file = path / ".audiolab_test"
            try:
                test_file.touch()
                test_file.unlink()
            except Exception as e:
                return Result.err(f"Model path {model_path} is not writable: {e}")

            return Result.ok(True)

        except Exception as e:
            return Result.err(f"Model path validation failed: {e}")

    @classmethod
    def get_installation_instructions(cls) -> Dict[str, List[str]]:
        """Get installation instructions for missing dependencies"""

        return {
            "required": [
                "pip install transformers>=4.35.0",
                "pip install torch>=1.11.0 torchaudio>=0.11.0",
                "pip install httpx>=0.25.0",
                "pip install pretty_midi>=0.2.10"
            ],
            "optional": [
                "pip install accelerate>=0.24.0  # Faster model loading",
                "pip install xformers>=0.0.22    # Memory-efficient attention",
                "pip install mido>=1.3.0         # Enhanced MIDI support",
                "pip install music21>=9.1.0      # Advanced music analysis"
            ],
            "gpu_support": [
                "# For NVIDIA GPU support:",
                "pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
            ],
            "soundraw": [
                "# Set SOUNDRAW API key in environment:",
                "export SOUNDRAW_API_KEY=your_api_key_here"
            ]
        }