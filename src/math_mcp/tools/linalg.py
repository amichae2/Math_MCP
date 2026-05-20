"""Linear algebra tools using NumPy and SciPy."""

import json
import time
from typing import Any

import numpy as np
from scipy import linalg
from scipy.sparse import linalg as sparse_linalg
import sympy

from ..utils.errors import tool_error_handler
from ..utils.latex_utils import safe_latex

MAX_DISPLAY_SIZE = 10


def _as_matrix(matrix: list[list[float]], *, name: str = "matrix") -> np.ndarray:
    if not matrix or not all(isinstance(row, list) and row for row in matrix):
        raise ValueError(f"{name} must be a non-empty rectangular list of rows")
    row_lengths = {len(row) for row in matrix}
    if len(row_lengths) != 1:
        raise ValueError(f"{name} must be rectangular")
    return np.asarray(matrix, dtype=float)


def _truncate_matrix(matrix: np.ndarray) -> list[list[float]]:
    return matrix[:MAX_DISPLAY_SIZE, :MAX_DISPLAY_SIZE].tolist()


def _truncate_vector(vector: np.ndarray) -> list[float]:
    return vector[:MAX_DISPLAY_SIZE].tolist()


def _relative_residual(a_matrix: np.ndarray, x_vector: np.ndarray, b_vector: np.ndarray) -> float:
    residual = np.linalg.norm(a_matrix @ x_vector - b_vector)
    denominator = np.linalg.norm(b_vector)
    return float(residual if denominator == 0 else residual / denominator)


def _is_symmetric(matrix: np.ndarray) -> bool:
    return matrix.shape[0] == matrix.shape[1] and np.allclose(matrix, matrix.T)


def _is_positive_definite(matrix: np.ndarray) -> bool:
    if not _is_symmetric(matrix):
        return False
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def _serialize_scalar(value: complex | float) -> float | str:
    if isinstance(value, complex) or np.iscomplexobj(value):
        complex_value = complex(value)
        if abs(complex_value.imag) < 1e-10:
            return float(complex_value.real)
        sign = "+" if complex_value.imag >= 0 else "-"
        return f"{complex_value.real:.12g}{sign}{abs(complex_value.imag):.12g}j"
    return float(value)


def _serialize_vector(values: np.ndarray) -> list[float | str]:
    return [_serialize_scalar(value) for value in values.tolist()]


@tool_error_handler("matrix_multiply")
async def matrix_multiply(
    a: list[list[float]],
    b: list[list[float]],
) -> dict[str, Any]:
    """Multiply two matrices A @ B."""
    a_matrix = _as_matrix(a, name="a")
    b_matrix = _as_matrix(b, name="b")
    if a_matrix.shape[1] != b_matrix.shape[0]:
        raise ValueError("matrix dimensions do not align: a_cols must equal b_rows")

    start_time = time.perf_counter()
    product = a_matrix @ b_matrix
    elapsed = time.perf_counter() - start_time
    response = {
        "result": f"{product.shape[0]}x{product.shape[1]} matrix computed",
        "product": _truncate_matrix(product),
        "shape": [int(product.shape[0]), int(product.shape[1])],
        "latex": None,
    }
    if max(a_matrix.shape + b_matrix.shape) > 100:
        response["time_taken_seconds"] = elapsed
    return response


@tool_error_handler("solve_linear")
async def solve_linear(
    a: list[list[float]],
    b: list[float],
    method: str = "auto",
) -> dict[str, Any]:
    """Solve a square linear system Ax = b."""
    a_matrix = _as_matrix(a, name="a")
    b_vector = np.asarray(b, dtype=float)
    if a_matrix.shape[0] != a_matrix.shape[1]:
        raise ValueError("a must be square")
    if a_matrix.shape[0] != b_vector.shape[0]:
        raise ValueError("b must have the same length as the size of a")

    condition_number = float(np.linalg.cond(a_matrix))
    method_name = method.lower()
    if method_name not in {"auto", "lu", "qr", "cholesky", "svd", "iterative"}:
        raise ValueError("method must be one of: auto, lu, qr, cholesky, svd, iterative")

    try:
        if method_name in {"auto", "lu"}:
            x_vector = np.linalg.solve(a_matrix, b_vector)
            method_used = "LU"
        elif method_name == "qr":
            q_matrix, r_matrix = np.linalg.qr(a_matrix)
            x_vector = linalg.solve_triangular(r_matrix, q_matrix.T @ b_vector)
            method_used = "QR"
        elif method_name == "cholesky":
            cholesky = np.linalg.cholesky(a_matrix)
            y_vector = linalg.solve_triangular(cholesky, b_vector, lower=True)
            x_vector = linalg.solve_triangular(cholesky.T, y_vector, lower=False)
            method_used = "Cholesky"
        elif method_name == "svd":
            x_vector, _, _, _ = linalg.lstsq(a_matrix, b_vector, lapack_driver="gelsd")
            method_used = "SVD"
        else:
            x_vector, info = sparse_linalg.cg(a_matrix, b_vector)
            if info != 0:
                raise np.linalg.LinAlgError("iterative solve did not converge")
            method_used = "Iterative"
    except np.linalg.LinAlgError as error:
        raise ValueError("matrix appears singular; try lstsq or regularization") from error

    residual_norm = _relative_residual(a_matrix, np.asarray(x_vector, dtype=float), b_vector)
    response = {
        "result": "Solution vector",
        "x": _truncate_vector(np.asarray(x_vector, dtype=float)),
        "residual_norm": residual_norm,
        "condition_number": condition_number,
        "method_used": method_used,
        "latex": None,
    }
    if condition_number > 1e10:
        response["warning"] = "matrix is ill-conditioned; the solution may be inaccurate"
    return response


@tool_error_handler("lstsq")
async def lstsq(
    a: list[list[float]],
    b: list[float],
    method: str = "gelsd",
    regularization: float = 0.0,
) -> dict[str, Any]:
    """Compute a least-squares solution to Ax ≈ b."""
    a_matrix = _as_matrix(a, name="a")
    b_vector = np.asarray(b, dtype=float)
    if a_matrix.shape[0] != b_vector.shape[0]:
        raise ValueError("b must have the same number of rows as a")
    if method not in {"gelsd", "gelss", "gelsy"}:
        raise ValueError("method must be one of: gelsd, gelss, gelsy")

    if regularization > 0:
        regularized_matrix = a_matrix.T @ a_matrix + regularization * np.eye(a_matrix.shape[1])
        regularized_rhs = a_matrix.T @ b_vector
        x_vector = np.linalg.solve(regularized_matrix, regularized_rhs)
        residual_vector = a_matrix @ x_vector - b_vector
        residuals = [float(np.linalg.norm(residual_vector))]
        rank = int(np.linalg.matrix_rank(a_matrix))
        singular_values = np.linalg.svd(a_matrix, compute_uv=False)
    else:
        x_vector, residual_values, rank, singular_values = linalg.lstsq(
            a_matrix,
            b_vector,
            lapack_driver=method,
        )
        residuals = np.asarray(residual_values, dtype=float).tolist() if np.size(residual_values) else [float(np.linalg.norm(a_matrix @ x_vector - b_vector))]

    return {
        "result": "Least-squares solution",
        "x": _truncate_vector(np.asarray(x_vector, dtype=float)),
        "residuals": residuals,
        "rank": int(rank),
        "singular_values": _truncate_vector(np.asarray(singular_values, dtype=float)),
        "latex": None,
    }


@tool_error_handler("matrix_decompose")
async def matrix_decompose(
    matrix: list[list[float]],
    decomposition: str = "all",
) -> dict[str, Any]:
    """Compute standard matrix decompositions."""
    input_matrix = _as_matrix(matrix)
    decomposition_name = decomposition.lower()
    if decomposition_name not in {"lu", "qr", "cholesky", "svd", "schur", "all"}:
        raise ValueError("decomposition must be one of: lu, qr, cholesky, svd, schur, all")

    results: dict[str, Any] = {"latex": None}
    rank = int(np.linalg.matrix_rank(input_matrix))

    if decomposition_name in {"lu", "all"}:
        p_matrix, l_matrix, u_matrix = linalg.lu(input_matrix)
        results["LU"] = {
            "P": _truncate_matrix(p_matrix),
            "L": _truncate_matrix(l_matrix),
            "U": _truncate_matrix(u_matrix),
            "rank": rank,
        }
    if decomposition_name in {"qr", "all"}:
        q_matrix, r_matrix = np.linalg.qr(input_matrix, mode="reduced")
        results["QR"] = {
            "Q": _truncate_matrix(q_matrix),
            "R": _truncate_matrix(r_matrix),
            "rank": rank,
        }
    if decomposition_name in {"cholesky", "all"} and input_matrix.shape[0] == input_matrix.shape[1] and _is_positive_definite(input_matrix):
        cholesky = np.linalg.cholesky(input_matrix)
        results["Cholesky"] = {
            "L": _truncate_matrix(cholesky),
            "rank": rank,
        }
    if decomposition_name in {"cholesky"} and "Cholesky" not in results:
        raise ValueError("matrix is not symmetric positive definite")
    if decomposition_name in {"svd", "all"}:
        u_matrix, singular_values, vh_matrix = np.linalg.svd(input_matrix, full_matrices=False)
        results["SVD"] = {
            "U": _truncate_matrix(u_matrix),
            "S": _truncate_vector(singular_values),
            "Vt": _truncate_matrix(vh_matrix),
            "rank": rank,
        }
    if decomposition_name in {"schur", "all"} and input_matrix.shape[0] == input_matrix.shape[1]:
        t_matrix, z_matrix = linalg.schur(input_matrix)
        results["Schur"] = {
            "T": _truncate_matrix(t_matrix),
            "Z": _truncate_matrix(z_matrix),
            "rank": rank,
        }
    return results


@tool_error_handler("eigen_decomp")
async def eigen_decomp(
    matrix: list[list[float]],
    compute_vectors: bool = True,
) -> dict[str, Any]:
    """Compute eigenvalues and optionally eigenvectors."""
    input_matrix = _as_matrix(matrix)
    if input_matrix.shape[0] != input_matrix.shape[1]:
        raise ValueError("matrix must be square")

    symmetric = _is_symmetric(input_matrix)
    if symmetric:
        eigenvalues, eigenvectors = np.linalg.eigh(input_matrix)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(input_matrix)

    sort_order = np.argsort(-np.abs(eigenvalues))
    eigenvalues = eigenvalues[sort_order]
    eigenvectors = eigenvectors[:, sort_order]

    grouped: dict[str, int] = {}
    for value in eigenvalues:
        key = str(np.round(value.real, 10)) if abs(value.imag) < 1e-10 else str(np.round(value, 10))
        grouped[key] = grouped.get(key, 0) + 1
    eigenvector_rank = int(np.linalg.matrix_rank(eigenvectors))

    response = {
        "result": "Eigenvalues: " + ", ".join(
            f"lambda_{index + 1}={_serialize_scalar(value)}" for index, value in enumerate(eigenvalues)
        ),
        "eigenvalues": _serialize_vector(eigenvalues),
        "algebraic_multiplicity": grouped,
        "is_diagonalizable": eigenvector_rank == input_matrix.shape[0],
        "condition_number_eigenvectors": float(np.linalg.cond(eigenvectors)),
        "latex": None,
    }
    if compute_vectors:
        response["eigenvectors"] = np.asarray(eigenvectors[:, :MAX_DISPLAY_SIZE]).T.tolist()
    return response


@tool_error_handler("matrix_info")
async def matrix_info(
    matrix: list[list[float]],
) -> dict[str, Any]:
    """Compute structural and numerical information about a matrix."""
    input_matrix = _as_matrix(matrix)
    square = input_matrix.shape[0] == input_matrix.shape[1]
    symmetric = _is_symmetric(input_matrix)
    eigenvalues = np.linalg.eigvalsh(input_matrix) if square and symmetric else None
    positive_definite = bool(eigenvalues is not None and np.all(eigenvalues > 1e-10))
    diagonal_dominant = bool(
        square
        and all(
            abs(input_matrix[index, index]) >= np.sum(np.abs(input_matrix[index])) - abs(input_matrix[index, index])
            for index in range(input_matrix.shape[0])
        )
    )

    result = {
        "result": f"{input_matrix.shape[0]}x{input_matrix.shape[1]} matrix, rank={int(np.linalg.matrix_rank(input_matrix))}"
        + (f", det={float(np.linalg.det(input_matrix))}" if square else ""),
        "shape": [int(input_matrix.shape[0]), int(input_matrix.shape[1])],
        "rank": int(np.linalg.matrix_rank(input_matrix)),
        "determinant": float(np.linalg.det(input_matrix)) if square else None,
        "trace": float(np.trace(input_matrix)) if square else None,
        "norm_1": float(np.linalg.norm(input_matrix, 1)),
        "norm_2": float(np.linalg.norm(input_matrix, 2)),
        "norm_inf": float(np.linalg.norm(input_matrix, np.inf)),
        "norm_frobenius": float(np.linalg.norm(input_matrix, "fro")),
        "condition_number": float(np.linalg.cond(input_matrix)) if square else None,
        "is_symmetric": symmetric,
        "is_positive_definite": positive_definite if square else None,
        "is_diagonally_dominant": diagonal_dominant if square else None,
        "sparsity": float(np.count_nonzero(np.isclose(input_matrix, 0.0)) / input_matrix.size),
        "latex": None,
    }
    return result


def register(server: Any) -> None:
    """Register all linear algebra tools."""
    server.tool("matrix_multiply")(matrix_multiply)
    server.tool("solve_linear")(solve_linear)
    server.tool("lstsq")(lstsq)
    server.tool("matrix_decompose")(matrix_decompose)
    server.tool("eigen_decomp")(eigen_decomp)
    server.tool("matrix_info")(matrix_info)