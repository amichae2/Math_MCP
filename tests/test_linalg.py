"""Tests for linear algebra tools."""

import numpy as np
import pytest

from math_mcp.tools.linalg import (
    eigen_decomp,
    lstsq,
    matrix_decompose,
    matrix_info,
    matrix_multiply,
    solve_linear,
)


# --- matrix_multiply ---


@pytest.mark.asyncio
async def test_matrix_multiply_2x2() -> None:
    result = await matrix_multiply(a=[[1.0, 2.0], [3.0, 4.0]], b=[[5.0, 6.0], [7.0, 8.0]])
    assert result["product"] == [[19.0, 22.0], [43.0, 50.0]]
    assert result["shape"] == [2, 2]


@pytest.mark.asyncio
async def test_matrix_multiply_incompatible() -> None:
    result = await matrix_multiply(a=[[1.0, 2.0]], b=[[3.0]])
    assert result["error"] is not None


# --- solve_linear ---


@pytest.mark.asyncio
async def test_solve_linear_2x2(sample_matrix) -> None:
    """Solve Ax=b for known SPD system."""
    result = await solve_linear(a=sample_matrix, b=[1.0, 2.0, 3.0])
    assert result["x"][0] == pytest.approx(0.24, rel=1e-6)
    assert result["x"][1] == pytest.approx(0.86, rel=1e-6)
    assert result["x"][2] == pytest.approx(0.82, rel=1e-6)
    assert result["residual_norm"] < 1e-10


@pytest.mark.asyncio
async def test_solve_linear_singular() -> None:
    result = await solve_linear(a=[[1.0, 2.0], [2.0, 4.0]], b=[3.0, 6.0])
    assert result["error"] is not None or "singular" in str(result.get("result", "")).lower()


# --- lstsq ---


@pytest.mark.asyncio
async def test_lstsq_overdetermined() -> None:
    result = await lstsq(a=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], b=[1.0, 2.0, 3.0])
    assert result["x"][0] == pytest.approx(1.0, rel=1e-6)
    assert result["x"][1] == pytest.approx(2.0, rel=1e-6)


# --- matrix_decompose ---


@pytest.mark.asyncio
async def test_decompose_svd(sample_matrix) -> None:
    result = await matrix_decompose(matrix=sample_matrix, decomposition="svd")
    assert "SVD" in result
    assert len(result["SVD"]["S"]) == 3


@pytest.mark.asyncio
async def test_decompose_lu() -> None:
    result = await matrix_decompose(matrix=[[2.0, 3.0], [1.0, 4.0]], decomposition="lu")
    assert "LU" in result
    assert "P" in result["LU"]
    assert "L" in result["LU"]
    assert "U" in result["LU"]


# --- eigen_decomp ---


@pytest.mark.asyncio
async def test_eigen_decomp_symmetric(sample_matrix) -> None:
    result = await eigen_decomp(matrix=sample_matrix, compute_vectors=True)
    assert "eigenvalues" in result
    assert "eigenvectors" in result
    assert len(result["eigenvalues"]) == 3


@pytest.mark.asyncio
async def test_eigen_decomp_2x2() -> None:
    result = await eigen_decomp(matrix=[[0.0, 1.0], [-1.0, 0.0]], compute_vectors=False)
    assert len(result["eigenvalues"]) == 2


# --- matrix_info ---


@pytest.mark.asyncio
async def test_matrix_info_square(sample_matrix) -> None:
    result = await matrix_info(matrix=sample_matrix)
    assert result["shape"] == [3, 3]
    assert result["rank"] == 3
    assert result["is_symmetric"] is True
    assert result["is_positive_definite"] is True


@pytest.mark.asyncio
async def test_matrix_info_rectangular() -> None:
    result = await matrix_info(matrix=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert result["shape"] == [2, 3]
    assert result["determinant"] is None
