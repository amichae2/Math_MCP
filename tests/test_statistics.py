"""Tests for statistics tools."""

import numpy as np
import pytest

from math_mcp.tools.statistics import (
    bootstrap,
    describe,
    distribution,
    hypothesis_test,
    random_sample,
    regression,
)


# --- describe ---


@pytest.mark.asyncio
async def test_describe_basic(sample_data) -> None:
    result = await describe(data=sample_data)
    assert result["n"] == 10
    assert result["mean"] == pytest.approx(5.52, rel=1e-4)
    assert "median" in result
    assert "std" in result


@pytest.mark.asyncio
async def test_describe_empty() -> None:
    result = await describe(data=[])
    assert result["error"] is not None


# --- distribution ---


@pytest.mark.asyncio
async def test_distribution_normal_pdf() -> None:
    result = await distribution(name="normal", operation="pdf", x=0.0)
    assert result["result"] == pytest.approx(0.398942, rel=1e-4)


@pytest.mark.asyncio
async def test_distribution_normal_cdf_ppf() -> None:
    result = await distribution(name="normal", operation="ppf", q=0.975)
    assert result["result"] == pytest.approx(1.96, rel=1e-2)


@pytest.mark.asyncio
async def test_distribution_rvs() -> None:
    result = await distribution(name="normal", operation="rvs", size=100, random_state=42)
    assert "sample_stats" in result
    assert result["sample_stats"]["mean"] == pytest.approx(0.0, abs=0.3)


@pytest.mark.asyncio
async def test_distribution_stats() -> None:
    result = await distribution(name="normal", operation="stats")
    assert result["mean"] == pytest.approx(0.0)
    assert result["variance"] == pytest.approx(1.0)


# --- hypothesis_test ---


@pytest.mark.asyncio
async def test_ttest_two_groups() -> None:
    """Two groups with clearly different means."""
    group1 = [5.0, 6.0, 7.0, 5.5, 6.5]
    group2 = [10.0, 11.0, 12.0, 10.5, 11.5]
    result = await hypothesis_test(test="ttest_ind", data1=group1, data2=group2)
    assert result["reject_h0"] is True
    assert result["p_value"] < 0.01


@pytest.mark.asyncio
async def test_ttest_same_groups() -> None:
    """Two groups with similar means."""
    group1 = [5.0, 6.0, 7.0, 5.5, 6.5]
    group2 = [5.1, 6.1, 6.9, 5.4, 6.6]
    result = await hypothesis_test(test="ttest_ind", data1=group1, data2=group2)
    assert result["reject_h0"] is False


@pytest.mark.asyncio
async def test_shapiro_normal() -> None:
    """Shapiro-Wilk on normal data should not reject."""
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, 100).tolist()
    result = await hypothesis_test(test="shapiro", data1=data)
    assert result["reject_h0"] is False


# --- regression ---


@pytest.mark.asyncio
async def test_regression_linear() -> None:
    """Perfect linear relationship y = 3x + 2."""
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [2.0, 5.0, 8.0, 11.0, 14.0]
    result = await regression(x=x, y=y, model="linear")
    assert result["coefficients"][0] == pytest.approx(2.0, rel=1e-6)
    assert result["coefficients"][1] == pytest.approx(3.0, rel=1e-6)
    assert result["r_squared"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_regression_polynomial() -> None:
    """Quadratic y = x squared."""
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    y = [0.0, 1.0, 4.0, 9.0, 16.0]
    result = await regression(x=x, y=y, model="polynomial", degree=2)
    assert result["r_squared"] == pytest.approx(1.0)


# --- bootstrap ---


@pytest.mark.asyncio
async def test_bootstrap_mean_ci() -> None:
    """Bootstrap CI for mean should be reasonable."""
    rng = np.random.default_rng(42)
    data = rng.normal(5.0, 2.0, 100).tolist()
    result = await bootstrap(data=data, n_resamples=1000)
    assert result["ci_lower"] < result["estimate"] < result["ci_upper"]
    assert result["estimate"] == pytest.approx(np.mean(data), rel=1e-6)


# --- random_sample ---


@pytest.mark.asyncio
async def test_random_sample_normal() -> None:
    result = await random_sample(distribution="normal", size=1000, random_state=42)
    assert len(result["samples"]) == 20
    assert result["sample_stats"]["mean"] == pytest.approx(0.0, abs=0.1)


@pytest.mark.asyncio
async def test_random_sample_uniform() -> None:
    result = await random_sample(distribution="uniform", params={"low": 0.0, "high": 10.0}, size=100, random_state=42)
    assert 0 <= min(result["samples"]) <= 10
    assert 0 <= max(result["samples"]) <= 10
