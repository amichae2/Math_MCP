"""Shared fixtures for math-mcp-server tests."""

import numpy as np
import pytest
import sympy


@pytest.fixture
def sample_expressions() -> dict[str, str]:
    """Common test expressions."""
    return {
        "linear": "2*x + 3",
        "quadratic": "x**2 - 4*x + 4",
        "trig": "sin(x)**2 + cos(x)**2",
        "rational": "(x**2 - 1)/(x - 1)",
        "exponential": "exp(-x**2)",
        "logarithmic": "log(x)",
    }


@pytest.fixture
def sample_matrix() -> list[list[float]]:
    """Standard test matrix."""
    return [[4.0, 1.0, -1.0], [1.0, 3.0, -1.0], [-1.0, -1.0, 5.0]]


@pytest.fixture
def sample_data() -> list[float]:
    """Standard test dataset."""
    return [1.0, 2.1, 2.9, 4.2, 5.0, 5.9, 7.1, 8.0, 8.9, 10.1]


@pytest.fixture
def numeric_tolerance() -> float:
    """Default tolerance for floating point comparisons."""
    return 1e-10


def assert_success(result: dict[str, object]) -> None:
    """Assert a standard successful tool response."""
    assert result.get("error") is None
