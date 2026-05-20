import pytest

from math_mcp.tools.rendering import plot_function, plot_implicit, render_math, to_latex


@pytest.mark.asyncio
async def test_to_latex_returns_latex_and_rendering() -> None:
    result = await to_latex("sin(x)/x")

    assert "sin" in result["latex"]
    assert result["rendered"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_render_math_returns_png_data_uri() -> None:
    result = await render_math(r"\int_0^1 x^2\,dx = \frac{1}{3}")

    assert result["data_uri"].startswith("data:image/png;base64,")
    assert result["width"] > 0
    assert result["height"] > 0


@pytest.mark.asyncio
async def test_plot_function_returns_plot_data_uri() -> None:
    result = await plot_function(["sin(x)", "cos(x)"], x_range=[-3.14, 3.14], num_points=200)

    assert result["data_uri"].startswith("data:image/png;base64,")
    assert result["x_range"] == [-3.14, 3.14]


@pytest.mark.asyncio
async def test_plot_implicit_returns_plot_data_uri() -> None:
    result = await plot_implicit("x**2 + y**2 = 1", x_range=[-2, 2], y_range=[-2, 2], grid_size=100)

    assert result["data_uri"].startswith("data:image/png;base64,")
    assert result["x_range"] == [-2.0, 2.0]
    assert result["y_range"] == [-2.0, 2.0]