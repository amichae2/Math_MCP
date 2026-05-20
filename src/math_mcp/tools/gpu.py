"""GPU-accelerated math tools using CuPy with NumPy fallback."""

import logging
import time
from typing import Any

import numpy as np

from ..config import HardwareConfig
from ..utils.errors import tool_error_handler

logger = logging.getLogger(__name__)

_hw_config: HardwareConfig | None = None

try:
    import cupy as cp

    _HAS_CUPY = True
    logger.info("CuPy loaded successfully for GPU acceleration")
except ImportError:
    _HAS_CUPY = False
    cp = None
    logger.info("CuPy not available - GPU tools will use CPU fallback")


def _get_gpu_mem_info() -> tuple[int, int] | None:
    if not _HAS_CUPY:
        return None
    try:
        return cp.cuda.runtime.memGetInfo()
    except Exception:
        return None


def _sample_matrix(matrix: np.ndarray) -> list[list[float]]:
    return matrix[:3, :3].tolist()


def _sample_array(values: np.ndarray) -> list[Any]:
    flat = np.asarray(values).reshape(-1)
    sample = flat[:10]
    if np.iscomplexobj(sample):
        return [complex(value).real if abs(complex(value).imag) < 1e-12 else f"{complex(value).real:.12g}{'+' if complex(value).imag >= 0 else '-'}{abs(complex(value).imag):.12g}j" for value in sample]
    return sample.tolist()


def _should_use_gpu(*shapes: tuple[int, ...], force_cpu: bool = False) -> tuple[bool, str | None]:
    if force_cpu or not _HAS_CUPY or _hw_config is None or not _hw_config.gpu_available:
        return False, None
    if any(max(shape) < 500 for shape in shapes):
        return False, None
    mem_info = _get_gpu_mem_info()
    if mem_info is None:
        return False, None
    free_bytes, _ = mem_info
    estimated = sum(int(np.prod(shape)) for shape in shapes) * 8 * 3
    if estimated > 0.8 * free_bytes:
        return False, "estimated GPU memory usage exceeds 80% of free memory; using CPU fallback"
    return True, None


@tool_error_handler("gpu_matrix_multiply")
async def gpu_matrix_multiply(
    a: list[list[float]],
    b: list[list[float]],
    force_cpu: bool = False,
) -> dict[str, Any]:
    """Multiply two matrices with GPU acceleration when beneficial."""
    a_matrix = np.asarray(a, dtype=float)
    b_matrix = np.asarray(b, dtype=float)
    if a_matrix.ndim != 2 or b_matrix.ndim != 2:
        raise ValueError("a and b must be rectangular 2D matrices")
    if a_matrix.shape[1] != b_matrix.shape[0]:
        raise ValueError("matrix dimensions do not align: a_cols must equal b_rows")

    use_gpu, warning = _should_use_gpu(a_matrix.shape, b_matrix.shape, force_cpu=force_cpu)
    if max(a_matrix.shape + b_matrix.shape) > 10000:
        warning = (warning + "; " if warning else "") + "consider chunking or out-of-core methods for very large matrices"

    if use_gpu:
        free_before, _ = _get_gpu_mem_info() or (0, 0)
        a_gpu = cp.asarray(a_matrix)
        b_gpu = cp.asarray(b_matrix)
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        product_gpu = a_gpu @ b_gpu
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - start
        product = cp.asnumpy(product_gpu)
        free_after, _ = _get_gpu_mem_info() or (free_before, 0)
        gpu_memory_used_mb = max(free_before - free_after, 0) / (1024**2)
        backend = "cupy"
    else:
        start = time.perf_counter()
        product = a_matrix @ b_matrix
        elapsed = time.perf_counter() - start
        gpu_memory_used_mb = 0.0
        backend = "numpy"

    response = {
        "result": f"{a_matrix.shape[0]}x{a_matrix.shape[1]} @ {b_matrix.shape[0]}x{b_matrix.shape[1]} = {product.shape[0]}x{product.shape[1]} matrix ({backend.upper()}, {elapsed:.4f}s)",
        "product_shape": [int(product.shape[0]), int(product.shape[1])],
        "product_sample": _sample_matrix(product),
        "backend": backend,
        "time_seconds": elapsed,
        "gpu_memory_used_mb": float(gpu_memory_used_mb),
        "latex": None,
    }
    if warning:
        response["warning"] = warning
    return response


@tool_error_handler("gpu_fft")
async def gpu_fft(
    data: list[float] | list[complex],
    operation: str = "fft",
    real_input: bool = True,
    axis: int = -1,
    norm: str = "backward",
) -> dict[str, Any]:
    """Compute FFTs on GPU when beneficial, otherwise on CPU."""
    array = np.asarray(data)
    if array.ndim not in {1, 2}:
        raise ValueError("data must be one-dimensional or two-dimensional")
    operation_name = operation.lower()
    valid_ops = {"fft", "ifft", "rfft", "irfft", "fft2", "ifft2"}
    if operation_name not in valid_ops:
        raise ValueError("operation must be one of: fft, ifft, rfft, irfft, fft2, ifft2")

    effective_operation = "rfft" if operation_name == "fft" and real_input and np.isrealobj(array) else operation_name
    use_gpu = _HAS_CUPY and _hw_config is not None and _hw_config.gpu_available and array.size > 10_000
    backend = "cupy" if use_gpu else "numpy"
    fft_module = cp.fft if use_gpu else np.fft
    input_array = cp.asarray(array) if use_gpu else array

    start = time.perf_counter()
    if effective_operation == "fft":
        output = fft_module.fft(input_array, axis=axis, norm=norm)
    elif effective_operation == "ifft":
        output = fft_module.ifft(input_array, axis=axis, norm=norm)
    elif effective_operation == "rfft":
        output = fft_module.rfft(input_array, axis=axis, norm=norm)
    elif effective_operation == "irfft":
        output = fft_module.irfft(input_array, axis=axis, norm=norm)
    elif effective_operation == "fft2":
        output = fft_module.fft2(input_array, norm=norm)
    else:
        output = fft_module.ifft2(input_array, norm=norm)
    if use_gpu:
        cp.cuda.Stream.null.synchronize()
        output_np = cp.asnumpy(output)
    else:
        output_np = np.asarray(output)
    elapsed = time.perf_counter() - start

    return {
        "result": f"{effective_operation.upper()} of {array.size} points ({backend.upper()}, {elapsed:.4f}s)",
        "output_sample": _sample_array(output_np),
        "output_length": int(output_np.size if output_np.ndim == 1 else output_np.shape[-1]),
        "backend": backend,
        "time_seconds": elapsed,
        "latex": None,
    }


@tool_error_handler("gpu_eigen_batch")
async def gpu_eigen_batch(
    matrices: list[list[list[float]]],
    symmetric: bool = True,
    eigenvalues_only: bool = False,
) -> dict[str, Any]:
    """Compute batched eigen decompositions with GPU fallback."""
    batch = np.asarray(matrices, dtype=float)
    if batch.ndim != 3:
        raise ValueError("matrices must be a batch of square matrices")
    batch_size, n_rows, n_cols = batch.shape
    if n_rows != n_cols:
        raise ValueError("all matrices must be square")

    use_gpu = _HAS_CUPY and _hw_config is not None and _hw_config.gpu_available and batch_size * n_rows > 1000
    backend = "cupy" if use_gpu else "numpy"
    start = time.perf_counter()
    if use_gpu:
        batch_gpu = cp.array(batch)
        if symmetric:
            if eigenvalues_only:
                eigenvalues = cp.linalg.eigvalsh(batch_gpu)
                eigenvectors = None
            else:
                eigenvalues, eigenvectors = cp.linalg.eigh(batch_gpu)
        else:
            if eigenvalues_only:
                eigenvalues = cp.linalg.eigvals(batch_gpu)
                eigenvectors = None
            else:
                eigenvalues, eigenvectors = cp.linalg.eig(batch_gpu)
        cp.cuda.Stream.null.synchronize()
        eigenvalues_np = cp.asnumpy(eigenvalues)
    else:
        eigenvalues_list = []
        eigenvectors = []
        for matrix in batch:
            if symmetric:
                values, vectors = np.linalg.eigh(matrix)
            else:
                values, vectors = np.linalg.eig(matrix)
            eigenvalues_list.append(values)
            if not eigenvalues_only:
                eigenvectors.append(vectors)
        eigenvalues_np = np.asarray(eigenvalues_list)
    elapsed = time.perf_counter() - start

    sample_eigenvectors = None
    if not eigenvalues_only and eigenvectors is not None:
        if use_gpu:
            eigenvectors_np_full = cp.asnumpy(eigenvectors)
        else:
            eigenvectors_np_full = np.asarray(eigenvectors)
        sample_eigenvectors = eigenvectors_np_full[:5, :, : min(5, eigenvectors_np_full.shape[-1])].tolist()

    return {
        "result": f"Eigenvalues for {batch_size} {n_rows}x{n_cols} matrices ({backend.upper()}, {elapsed:.4f}s)",
        "batch_size": int(batch_size),
        "matrix_size": int(n_rows),
        "eigenvalues": [row.tolist() for row in eigenvalues_np[:5]],
        "eigenvectors": sample_eigenvectors,
        "backend": backend,
        "time_seconds": elapsed,
        "stats": {
            "min_eigenvalue": float(np.min(np.real(eigenvalues_np))),
            "max_eigenvalue": float(np.max(np.real(eigenvalues_np))),
            "mean_eigenvalue": float(np.mean(np.real(eigenvalues_np))),
        },
        "latex": None,
    }


@tool_error_handler("gpu_solve")
async def gpu_solve(
    a_batch: list[list[list[float]]],
    b_batch: list[list[float]] | list[list[list[float]]],
) -> dict[str, Any]:
    """Solve batched linear systems with GPU fallback."""
    a_array = np.asarray(a_batch, dtype=float)
    b_array = np.asarray(b_batch, dtype=float)
    if a_array.ndim != 3:
        raise ValueError("a_batch must have shape [batch, n, n]")
    batch_size, n_rows, n_cols = a_array.shape
    if n_rows != n_cols:
        raise ValueError("coefficient matrices must be square")
    if b_array.shape[0] != batch_size or b_array.shape[1] != n_rows:
        raise ValueError("b_batch dimensions must match a_batch")

    use_gpu = _HAS_CUPY and _hw_config is not None and _hw_config.gpu_available and batch_size * n_rows > 1000
    backend = "cupy" if use_gpu else "numpy"
    start = time.perf_counter()
    if use_gpu:
        chunk_size = 1000 if batch_size > 1000 else batch_size
        solutions = []
        for start_index in range(0, batch_size, chunk_size):
            stop_index = min(start_index + chunk_size, batch_size)
            a_gpu = cp.asarray(a_array[start_index:stop_index])
            b_gpu = cp.asarray(b_array[start_index:stop_index])
            solved = cp.linalg.solve(a_gpu, b_gpu)
            solutions.append(cp.asnumpy(solved))
        solution_array = np.concatenate(solutions, axis=0)
    else:
        solved = [np.linalg.solve(a_array[index], b_array[index]) for index in range(batch_size)]
        solution_array = np.asarray(solved)
    elapsed = time.perf_counter() - start

    residuals = []
    for index in range(min(3, batch_size)):
        residual = np.linalg.norm(a_array[index] @ solution_array[index] - b_array[index])
        residuals.append(float(residual))

    return {
        "result": f"Solved {batch_size} {n_rows}x{n_cols} systems ({backend.upper()}, {elapsed:.4f}s)",
        "solutions_sample": solution_array[:3].tolist(),
        "batch_size": int(batch_size),
        "backend": backend,
        "time_seconds": elapsed,
        "residuals": residuals,
        "latex": None,
    }


def register(server: Any, hw_config: HardwareConfig) -> None:
    """Register all GPU tools, providing hardware config."""
    global _hw_config
    _hw_config = hw_config

    server.tool("gpu_matrix_multiply")(gpu_matrix_multiply)
    server.tool("gpu_fft")(gpu_fft)
    server.tool("gpu_eigen_batch")(gpu_eigen_batch)
    server.tool("gpu_solve")(gpu_solve)