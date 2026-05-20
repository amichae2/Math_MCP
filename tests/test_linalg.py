import pytest
import numpy as np

from math_mcp.tools.linalg import (
    eigen_decomp,
    lstsq,
    matrix_decompose,
    matrix_info,
    matrix_multiply,
    solve_linear,
)


@pytest.mark.asyncio
async def test_matrix_multiply_returns_expected_product() -> None:
    result = await matrix_multiply([[1.0, 2.0], [3.0, 4.0]], [[2.0], [1.0]])

    assert result["shape"] == [2, 1]
    assert result["product"] == [[4.0], [10.0]]


@pytest.mark.asyncio
async def test_solve_linear_returns_expected_solution() -> None:
    result = await solve_linear([[3.0, 1.0], [1.0, 2.0]], [9.0, 8.0])

    assert result["x"] == pytest.approx([2.0, 3.0])
    assert result["method_used"] == "LU"


@pytest.mark.asyncio
async def test_solve_linear_3x3_system(sample_matrix) -> None:
    right_hand_side = [4.0, 3.0, 7.0]
    result = await solve_linear(sample_matrix, right_hand_side)

    assert np.allclose(np.asarray(sample_matrix) @ np.asarray(result["x"]), right_hand_side)


@pytest.mark.asyncio
async def test_solve_linear_singular_matrix_error() -> None:
    result = await solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])

    assert result["error"] is not None


@pytest.mark.asyncio
async def test_solve_linear_ill_conditioned_warning() -> None:
    hilbert = [[1.0 / (row + col + 1) for col in range(12)] for row in range(12)]
    rhs = [sum(row) for row in hilbert]
    result = await solve_linear(hilbert, rhs)

    assert result.get("warning") is not None


@pytest.mark.asyncio
async def test_solve_linear_iterative_returns_expected_solution() -> None:
    result = await solve_linear([[4.0, 1.0], [1.0, 3.0]], [1.0, 2.0], method="iterative")

    assert result["x"] == pytest.approx([0.0909090909, 0.6363636364], rel=1e-7)
    assert result["method_used"] == "Iterative"


@pytest.mark.asyncio
async def test_lstsq_returns_minimum_norm_solution() -> None:
    result = await lstsq([[1.0, 1.0], [1.0, -1.0], [1.0, 2.0]], [2.0, 0.0, 3.0])

    assert result["rank"] == 2
    assert result["x"] == pytest.approx([1.0, 1.0])


@pytest.mark.asyncio
async def test_lstsq_regularized_solution() -> None:
    result = await lstsq([[1.0, 1.0], [1.0, 1.001], [1.0, 0.999]], [2.0, 2.001, 1.999], regularization=1e-3)

    assert result["rank"] == 2
    assert len(result["x"]) == 2


@pytest.mark.asyncio
async def test_matrix_decompose_returns_qr_and_svd() -> None:
    result = await matrix_decompose([[1.0, 2.0], [3.0, 4.0]], decomposition="all")

    assert "QR" in result
    assert "SVD" in result


@pytest.mark.asyncio
async def test_matrix_decompose_svd_reconstructs_matrix() -> None:
    original = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    result = await matrix_decompose(original.tolist(), decomposition="svd")

    u_matrix = np.asarray(result["SVD"]["U"])
    singular_values = np.asarray(result["SVD"]["S"])
    vt_matrix = np.asarray(result["SVD"]["Vt"])
    reconstructed = u_matrix @ np.diag(singular_values) @ vt_matrix
    assert np.allclose(reconstructed, original)


@pytest.mark.asyncio
async def test_eigen_decomp_returns_sorted_eigenvalues() -> None:
    result = await eigen_decomp([[2.0, 0.0], [0.0, 1.0]])

    assert result["eigenvalues"] == pytest.approx([2.0, 1.0])
    assert result["is_diagonalizable"] is True


@pytest.mark.asyncio
async def test_eigen_decomp_non_symmetric_complex_case() -> None:
    result = await eigen_decomp([[0.0, -1.0], [1.0, 0.0]])

    assert any("j" in str(value) for value in result["eigenvalues"])


@pytest.mark.asyncio
async def test_matrix_info_reports_basic_properties() -> None:
    result = await matrix_info([[2.0, 0.0], [0.0, 3.0]])

    assert result["shape"] == [2, 2]
    assert result["rank"] == 2
    assert result["determinant"] == pytest.approx(6.0)
    assert result["is_positive_definite"] is True


@pytest.mark.asyncio
async def test_matrix_info_non_square_skips_square_only_fields() -> None:
    result = await matrix_info([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert result["determinant"] is None
    assert result["is_positive_definite"] is None