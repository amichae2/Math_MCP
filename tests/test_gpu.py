from math import isclose

import pytest

from math_mcp.config import HardwareConfig
from math_mcp.tools import gpu


class _DummyServer:
    def tool(self, name: str):
        def decorator(func):
            return func

        return decorator


gpu.register(_DummyServer(), HardwareConfig(cpu_cores=4, gpu_available=False, use_gpu_by_default=False))


@pytest.mark.asyncio
async def test_gpu_matrix_multiply_falls_back_to_cpu() -> None:
    result = await gpu.gpu_matrix_multiply([[1.0, 2.0], [3.0, 4.0]], [[2.0], [1.0]])

    assert result["backend"] == "numpy"
    assert result["product_shape"] == [2, 1]
    assert result["product_sample"] == [[4.0], [10.0]]


@pytest.mark.asyncio
async def test_gpu_fft_returns_numpy_backend_sample() -> None:
    result = await gpu.gpu_fft([0.0, 1.0, 0.0, -1.0], operation="fft", real_input=False)

    assert result["backend"] == "numpy"
    assert result["output_length"] == 4


@pytest.mark.asyncio
async def test_gpu_eigen_batch_returns_batch_stats() -> None:
    result = await gpu.gpu_eigen_batch([[[2.0, 0.0], [0.0, 1.0]], [[3.0, 0.0], [0.0, 4.0]]])

    assert result["batch_size"] == 2
    assert isclose(result["stats"]["max_eigenvalue"], 4.0)


@pytest.mark.asyncio
async def test_gpu_solve_returns_residuals() -> None:
    result = await gpu.gpu_solve(
        [[[2.0, 0.0], [0.0, 4.0]], [[1.0, 1.0], [1.0, -1.0]]],
        [[2.0, 8.0], [4.0, 0.0]],
    )

    assert result["backend"] == "numpy"
    assert result["batch_size"] == 2
    assert max(result["residuals"]) < 1e-9