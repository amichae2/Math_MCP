import math

import numpy as np
import pytest

from math_mcp.tools.numerical import curve_fit, find_root, interpolate_data, numerical_integrate, ode_solve


@pytest.mark.asyncio
async def test_numerical_integrate_polynomial() -> None:
    result = await numerical_integrate("x**2", lower=0.0, upper=1.0)

    assert result["result"] == pytest.approx(1 / 3, rel=1e-10)


@pytest.mark.asyncio
async def test_numerical_integrate_improper_integral() -> None:
    result = await numerical_integrate("exp(-x)", lower=0.0, upper=np.inf)

    assert result["result"] == pytest.approx(1.0, rel=1e-8)


@pytest.mark.asyncio
async def test_numerical_integrate_sine() -> None:
    result = await numerical_integrate("sin(x)", lower=0.0, upper=math.pi)

    assert result["result"] == pytest.approx(2.0, rel=1e-10)


@pytest.mark.asyncio
async def test_numerical_integrate_divergence_error() -> None:
    result = await numerical_integrate("1/x", lower=0.0, upper=1.0, method="romberg")

    assert result["method_used"] == "romberg"
    assert result["error_estimate"] > 1.0


@pytest.mark.asyncio
async def test_find_root_cos_minus_x() -> None:
    result = await find_root("cos(x) - x", bracket=[0.0, 1.0])

    assert result["result"] == pytest.approx(0.7390851332, rel=1e-9)
    assert result["converged"] is True


@pytest.mark.asyncio
async def test_find_root_quadratic_bracket() -> None:
    result = await find_root("x**2 - 4", bracket=[0.0, 3.0])

    assert result["result"] == pytest.approx(2.0, rel=1e-10)


@pytest.mark.asyncio
async def test_find_root_newton_on_cubic() -> None:
    result = await find_root("x**3", guess=0.1, method="newton")

    assert result["result"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.asyncio
async def test_find_root_no_root_case() -> None:
    result = await find_root("x**2 + 1")

    assert result["error"] is not None


@pytest.mark.asyncio
async def test_ode_solve_exponential_decay() -> None:
    result = await ode_solve("-y", y0=[1.0], t_span=[0.0, 1.0], num_points=25)

    assert result["success"] is True
    assert result["y_values"][0][-1] == pytest.approx(math.exp(-1), rel=1e-3)


@pytest.mark.asyncio
async def test_ode_solve_harmonic_oscillator() -> None:
    result = await ode_solve("y[1], -y[0]", y0=[0.0, 1.0], t_span=[0.0, math.pi / 2], num_points=50)

    assert result["success"] is True
    assert result["y_values"][0][-1] == pytest.approx(1.0, rel=1e-3)


@pytest.mark.asyncio
async def test_ode_solve_stiff_method() -> None:
    result = await ode_solve("-15*y", y0=[1.0], t_span=[0.0, 1.0], method="BDF", num_points=20)

    assert result["success"] is True


@pytest.mark.asyncio
async def test_interpolate_linear_matches_known_points() -> None:
    result = await interpolate_data([0.0, 1.0, 2.0], [0.0, 1.0, 4.0], method="linear", x_eval=[0.0, 1.0, 2.0])

    assert result["y_eval"] == pytest.approx([0.0, 1.0, 4.0])


@pytest.mark.asyncio
async def test_interpolate_cubic_matches_known_points() -> None:
    result = await interpolate_data([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 8.0, 27.0], method="cubic_spline", x_eval=[0.0, 1.0, 2.0, 3.0])

    assert result["y_eval"] == pytest.approx([0.0, 1.0, 8.0, 27.0])


@pytest.mark.asyncio
async def test_curve_fit_linear_parameters() -> None:
    result = await curve_fit("a*x + b", [0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0], parameters=["a", "b"])

    assert result["parameters"]["a"] == pytest.approx(2.0, rel=1e-6)
    assert result["parameters"]["b"] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.asyncio
async def test_curve_fit_exponential_model() -> None:
    x_data = [0.0, 1.0, 2.0, 3.0]
    y_data = [3.0, 3.0 * math.exp(-0.5), 3.0 * math.exp(-1.0), 3.0 * math.exp(-1.5)]
    result = await curve_fit("a*exp(-b*x)", x_data, y_data, parameters=["a", "b"], p0=[2.5, 0.4])

    assert result["parameters"]["a"] == pytest.approx(3.0, rel=1e-2)
    assert result["parameters"]["b"] == pytest.approx(0.5, rel=1e-2)