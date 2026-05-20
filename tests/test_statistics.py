import pytest
import math

from math_mcp.tools.statistics import (
    bootstrap,
    describe,
    distribution,
    hypothesis_test,
    random_sample,
    regression,
)


@pytest.mark.asyncio
async def test_describe_returns_expected_summary() -> None:
    result = await describe([1.0, 2.0, 3.0, 4.0])

    assert result["n"] == 4
    assert result["mean"] == pytest.approx(2.5)
    assert result["median"] == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_describe_known_data(sample_data) -> None:
    result = await describe(sample_data)

    assert result["n"] == len(sample_data)
    assert result["mean"] == pytest.approx(sum(sample_data) / len(sample_data))
    assert result["std"] > 0


@pytest.mark.asyncio
async def test_distribution_normal_cdf_returns_expected_value() -> None:
    result = await distribution("normal", "cdf", params={"loc": 0.0, "scale": 1.0}, x=0.0)

    assert result["result"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_distribution_normal_pdf_at_zero() -> None:
    result = await distribution("normal", "pdf", params={"loc": 0.0, "scale": 1.0}, x=0.0)

    assert result["result"] == pytest.approx(1 / math.sqrt(2 * math.pi), rel=1e-10)


@pytest.mark.asyncio
async def test_distribution_ppf_returns_expected_quantile() -> None:
    result = await distribution("normal", "ppf", params={"loc": 0.0, "scale": 1.0}, q=0.975)

    assert result["result"] == pytest.approx(1.9599639845, rel=1e-6)


@pytest.mark.asyncio
async def test_hypothesis_test_ttest_one_sample_rejects_zero_mean() -> None:
    result = await hypothesis_test("ttest_1samp", [2.0, 2.5, 3.0, 3.5], params={"popmean": 0.0})

    assert result["reject_h0"] is True
    assert result["statistic"] > 0


@pytest.mark.asyncio
async def test_hypothesis_test_two_sample_difference() -> None:
    result = await hypothesis_test("ttest_ind", [10.0, 11.0, 12.0, 13.0], [1.0, 2.0, 3.0, 4.0])

    assert result["reject_h0"] is True
    assert result["effect_size"] is not None


@pytest.mark.asyncio
async def test_regression_linear_returns_expected_coefficients() -> None:
    result = await regression([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])

    assert result["coefficients"] == pytest.approx([1.0, 2.0])
    assert result["r_squared"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_regression_linear_with_noise() -> None:
    x_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_values = [1.1, 3.0, 5.2, 7.0, 9.1]
    result = await regression(x_values, y_values)

    assert result["coefficients"][0] == pytest.approx(1.0, abs=0.3)
    assert result["coefficients"][1] == pytest.approx(2.0, abs=0.2)


@pytest.mark.asyncio
async def test_regression_polynomial() -> None:
    x_values = [0.0, 1.0, 2.0, 3.0]
    y_values = [1.0, 2.0, 5.0, 10.0]
    result = await regression(x_values, y_values, model="polynomial", degree=2)

    assert len(result["coefficients"]) == 3


@pytest.mark.asyncio
async def test_bootstrap_returns_confidence_interval() -> None:
    result = await bootstrap([1.0, 2.0, 3.0, 4.0], n_resamples=2000, random_state=7)

    assert result["estimate"] == pytest.approx(2.5)
    assert result["ci_lower"] <= result["estimate"] <= result["ci_upper"]


@pytest.mark.asyncio
async def test_bootstrap_constant_data() -> None:
    result = await bootstrap([2.0, 2.0, 2.0, 2.0], n_resamples=500, random_state=3)

    assert result["estimate"] == pytest.approx(2.0)
    assert result.get("warning") is not None


@pytest.mark.asyncio
async def test_random_sample_returns_stats_and_samples() -> None:
    result = await random_sample("normal", params={"loc": 0.0, "scale": 1.0}, size=100, random_state=11)

    assert len(result["samples"]) <= 20
    assert "mean" in result["sample_stats"]


@pytest.mark.asyncio
async def test_random_sample_multivariate_normal() -> None:
    result = await random_sample(
        "multivariate_normal",
        params={"mean": [0.0, 0.0], "cov": [[1.0, 0.0], [0.0, 1.0]]},
        size=10,
        dimensions=2,
        random_state=5,
    )

    assert len(result["samples"]) <= 20
    assert result["sample_stats"]["std"] > 0