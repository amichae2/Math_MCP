"""Hardware detection and configuration for math-mcp-server."""

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HardwareConfig:
    cpu_cores: int = 1
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    max_matrix_size_gpu: int = 1000
    max_symbolic_timeout: float = 10.0
    use_gpu_by_default: bool = False

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return {
            "cpu_cores": self.cpu_cores,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_memory_gb": round(self.gpu_memory_gb, 2),
            "max_matrix_size_gpu": self.max_matrix_size_gpu,
        }


def detect_hardware() -> HardwareConfig:
    """
    Detect available hardware resources.

    For GPU: tries CuPy first, then checks for CUDA toolkit.
    Calculates safe max matrix size based on available VRAM
    (leaving 20% headroom for intermediates).
    """
    config = HardwareConfig()

    config.cpu_cores = os.cpu_count() or 1

    try:
        import cupy as cp

        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        config.gpu_available = True
        config.gpu_memory_gb = total_bytes / (1024**3)

        try:
            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            config.gpu_name = name.decode() if isinstance(name, bytes) else name
        except Exception:
            config.gpu_name = "Unknown NVIDIA GPU"

        safe_bytes = free_bytes * 0.8
        config.max_matrix_size_gpu = int((safe_bytes / (3 * 8)) ** 0.5)
        config.use_gpu_by_default = True

        logger.info(
            "GPU detected: %s (%.1f GB)",
            config.gpu_name,
            config.gpu_memory_gb,
        )
        logger.info(
            "Max safe GPU matrix size: %sx%s",
            config.max_matrix_size_gpu,
            config.max_matrix_size_gpu,
        )
    except ImportError:
        logger.info("CuPy not installed - GPU acceleration unavailable")
    except Exception as error:
        logger.warning("GPU detection failed: %s", error)

    return config


hw_config = detect_hardware()