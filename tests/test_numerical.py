"""Tests for numerical computation tools."""

import numpy as np
import pytest

from math_mcp.tools.numerical import (
    curve_fit,
    find_root,
    interpolate_data,
    numerical_integrate,
    ode_solve,
)


# --- numerical_integrate ---


@pytest.mark.asyncio
async def test_integrate_x_squared() -> None:
    """Integral from 0 to 1 of x squared equals 1/3."""
    result = await numerical_integrate(expression="x**2", lower=0.0, upper=1.0)
    assert result["result"] == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert result["method_used"] in {"quad", "romberg"}


@pytest.mark.asyncio
async def test_integrate_exp_negative_x() -> None:
    """Integral from 0 to infinity of exp(-x) equals 1."""
    result = await numerical_integrate(expression="exp(-x)", lower=0.0, upper=float("inf"))
    assert result["result"] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.asyncio
async def test_integrate_sin() -> None:
    """Integral from 0 to pi of sin(x) equals 2."""
    result = await numerical_integrate(expression="sin(x)", lower=0.0, upper=np.pi)
    assert result["result"] == pytest.approx(2.0, rel=1e-6)


@pytest.mark.asyncio
async def test_integrate_auto_break_points() -> None:
    """Integral from -1 to 1 of Abs(x) equals 1 and should detect x=0."""
    result = await numerical_integrate(expression="Abs(x)", lower=-1.0, upper=1.0)
    assert result["result"] == pytest.approx(1.0, rel=1e-6)


@pytest.mark.asyncio
async def test_integrate_with_simpson() -> None:
    """Integral from 0 to 1 of x squared using Simpson's rule."""
    result = await numerical_integrate(expression="x**2", lower=0.0, upper=1.0, method="simpson")
    assert result["result"] == pytest.approx(1.0 / 3.0, rel=1e-4)


@pytest.mark.asyncio
async def test_integrate_invalid_method() -> None:
    result = await numerical_integrate(expression="x", method="nonexistent")
    assert result["error"] is not None


# --- find_root ---


@pytest.mark.asyncio
async def test_find_root_quadratic() -> None:
    """Root of x squared minus 4 equals 0 in [0, 3]."""
    result = await find_root(expression="x**2 - 4", bracket=[0.0, 3.0])
    assert result["result"] == pytest.approx(2.0)
    assert result["converged"] is True


@pytest.mark.asyncio
async def test_find_root_cos_minus_x() -> None:
    """Root of cos(x) - x is the Dottie number."""
    result = await find_root(expression="cos(x) - x", bracket=[0.0, 1.0])
    assert result["result"] == pytest.approx(0.739085, rel=1e-4)


@pytest.mark.asyncio
async def test_find_root_with_guess_newton() -> None:
    """Root of x cubed minus 2 near x=1.26."""
    result = await find_root(expression="x**3 - 2", guess=1.5, method="newton")
    assert result["result"] == pytest.approx(2.0 ** (1.0 / 3.0), rel=1e-6)


@pytest.mark.asyncio
async def test_find_root_constant_function() -> None:
    """Root finding on constant function should be handled gracefully."""
    result = await find_root(expression="5", bracket=[0.0, 1.0])
    assert "constant" in str(result.get("result", "")).lower() or result["error"] is not None


# --- ode_solve ---


@pytest.mark.asyncio
async def test_ode_exponential_decay() -> None:
    """y' = -y, y(0)=1 gives y(t)=exp(-t)."""
    result = await ode_solve(dydt="-y", y0=[1.0], t_span=[0.0, 1.0], num_points=10)
    assert result["success"] is True
    y_final = result["y_values"][0][-1] if isinstance(result["y_values"][0], list) else result["y_values"][-1]
    y_final = y_final if isinstance(y_final, float) else y_final[0]
    assert y_final == pytest.approx(0.3679, rel=1e-2)


@pytest.mark.asyncio
async def test_ode_harmonic_oscillator() -> None:
    """y'' = -y, y(0)=0, y'(0)=1 gives y(t)=sin(t)."""
    result = await ode_solve(dydt="y[1], -y[0]", y0=[0.0, 1.0], t_span=[0.0, np.pi / 2], num_points=20)
    assert result["success"] is True
    y_final = result["y_values"][0][-1] if isinstance(result["y_values"][0], list) else result["y_values"][-1]
    y_final = y_final if isinstance(y_final, float) else y_final[0]
    assert y_final == pytest.approx(1.0, rel=1e-2)


# --- interpolate_data ---


@pytest.mark.asyncio
async def test_interpolate_linear() -> None:
    result = await interpolate_data(
        x_data=[0.0, 1.0, 2.0],
        y_data=[0.0, 1.0, 4.0],
        method="linear",
        x_eval=[0.5, 1.5],
    )
    assert result["y_eval"] == pytest.approx([0.5, 2.5], rel=1e-6)


@pytest.mark.asyncio
async def test_interpolate_cubic_spline() -> None:
    result = await interpolate_data(
        x_data=[0.0, 1.0, 2.0, 3.0],
        y_data=[0.0, 1.0, 4.0, 9.0],
        method="cubic_spline",
        x_eval=[1.0, 2.0],
    )
    assert result["y_eval"] == pytest.approx([1.0, 4.0], rel=1e-6)


@pytest.mark.asyncio
async def test_interpolate_insufficient_points() -> None:
    result = await interpolate_data(
        x_data=[0.0],
        y_data=[0.0],
        method="cubic_spline",
    )
    assert result["error"] is not None


# --- curve_fit ---


@pytest.mark.asyncio
async def test_curve_fit_exponential_decay() -> None:
    """Fit a*exp(-b*x) + c to clean data."""
    x = [0.0, 0.5, 1.0, 1.5, 2.0]
    y = [3.0, 2.2130613194, 1.7357588823, 1.4462603203, 1.2706705665]
    result = await curve_fit(model="a*exp(-b*x) + c", x_data=x, y_data=y, p0=[2.0, 1.0, 1.0])
    assert result["parameters"]["a"] == pytest.approx(2.0, rel=0.1)
    assert result["parameters"]["c"] == pytest.approx(1.0, rel=0.1)
    assert result["r_squared"] > 0.99


@pytest.mark.asyncio
async def test_curve_fit_linear() -> None:
    """Fit a*x + b to a perfect line."""
    x = [0.0, 1.0, 2.0, 3.0]
    y = [2.0, 5.0, 8.0, 11.0]
    result = await curve_fit(model="a*x + b", x_data=x, y_data=y, parameters=["a", "b"])
    assert result["parameters"]["a"] == pytest.approx(3.0, rel=1e-6)
    assert result["parameters"]["b"] == pytest.approx(2.0, rel=1e-6)
    assert result["r_squared"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_curve_fit_too_many_parameters() -> None:
    result = await curve_fit(model="a*x + b", x_data=[0.0], y_data=[1.0])
    assert result["error"] is not None
