"""Rendering and visualization tools using matplotlib and SymPy."""

import base64
import io
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sympy
from matplotlib.backends.backend_agg import FigureCanvasAgg
from sympy.core.relational import Relational

from ..utils.parsing import parse_expression
from ..utils.errors import tool_error_handler
from ..utils.latex_utils import safe_latex, latex_to_image_datauri

logger = logging.getLogger(__name__)


def _figure_to_data_uri(fig: plt.Figure, *, dpi: int = 150) -> tuple[str, int, int]:
    buffer = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode()
    return f"data:image/png;base64,{encoded}", int(width), int(height)


def _plot_style(dark_mode: bool) -> dict[str, str]:
    if dark_mode:
        return {
            "facecolor": "#111111",
            "axes_facecolor": "#111111",
            "grid_color": "#666666",
            "text_color": "#f5f5f5",
        }
    return {
        "facecolor": "#ffffff",
        "axes_facecolor": "#ffffff",
        "grid_color": "#dddddd",
        "text_color": "#111111",
    }


def _auto_x_range(expressions: list[sympy.Expr]) -> list[float]:
    if any(expr.has(sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc) for expr in expressions):
        return [-2 * float(sympy.pi), 2 * float(sympy.pi)]
    if any(expr.is_rational_function() for expr in expressions):
        return [-10.0, 10.0]
    if any(expr.has(sympy.exp) for expr in expressions):
        return [-3.0, 3.0]
    return [-10.0, 10.0]


def _sanitize_curve(x_values: np.ndarray, y_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_values = np.asarray(y_values, dtype=float)
    finite_mask = np.isfinite(y_values)
    if not np.any(finite_mask):
        return x_values, np.full_like(x_values, np.nan, dtype=float)
    clean_values = y_values.copy()
    clean_values[~finite_mask] = np.nan
    differences = np.abs(np.diff(clean_values))
    finite_differences = differences[np.isfinite(differences)]
    if finite_differences.size:
        threshold = max(np.percentile(finite_differences, 95) * 10, 1e3)
        jump_indices = np.where(differences > threshold)[0]
        clean_values[jump_indices + 1] = np.nan
    return x_values, clean_values


def _parse_implicit_equation(equation: str) -> tuple[sympy.Basic, str | None]:
    for operator in ("<=", ">=", "<", ">", "="):
        if operator in equation:
            left, right = equation.split(operator, 1)
            left_expr = parse_expression(left)
            right_expr = parse_expression(right)
            if operator == "=":
                return sympy.simplify(left_expr - right_expr), None
            relation = {
                "<=": sympy.Le,
                ">=": sympy.Ge,
                "<": sympy.Lt,
                ">": sympy.Gt,
            }[operator](left_expr - right_expr, 0)
            return relation, operator
    return parse_expression(equation), None


@tool_error_handler("to_latex")
async def to_latex(
    expression: str,
    style: str = "inline",
) -> dict[str, Any]:
    """Convert a mathematical expression to LaTeX and render it."""
    expr = parse_expression(expression)
    normalized_style = style.lower()
    if normalized_style == "inline":
        latex = sympy.latex(expr, mul_symbol="dot", fold_short_frac=True)
    elif normalized_style == "display":
        latex = sympy.latex(expr, mul_symbol="times")
    elif normalized_style == "plain":
        latex = safe_latex(expr)
    else:
        raise ValueError("style must be one of: inline, display, plain")
    rendered = latex_to_image_datauri(latex)
    return {
        "result": latex,
        "latex": latex,
        "rendered": rendered,
        "style": normalized_style,
    }


@tool_error_handler("render_math")
async def render_math(
    latex: str,
    fontsize: int = 16,
    dpi: int = 150,
) -> dict[str, Any]:
    """Render arbitrary LaTeX math to a PNG data URI."""
    data_uri = latex_to_image_datauri(latex, fontsize=fontsize, dpi=dpi)
    if data_uri is None:
        return {
            "result": "Rendering failed; returning raw LaTeX",
            "data_uri": None,
            "width": None,
            "height": None,
            "latex": latex,
            "warning": "matplotlib mathtext could not render the provided LaTeX",
        }

    image_bytes = base64.b64decode(data_uri.split(",", 1)[1])
    image = plt.imread(io.BytesIO(image_bytes), format="png")
    return {
        "result": f"Rendered LaTeX ({fontsize}pt, {dpi}dpi)",
        "data_uri": data_uri,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "latex": latex,
    }


@tool_error_handler("plot_function")
async def plot_function(
    expression: str | list[str],
    variable: str = "x",
    x_range: list[float] | None = None,
    num_points: int = 500,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    grid: bool = True,
    dark_mode: bool = False,
    figsize: list[int] | None = None,
) -> dict[str, Any]:
    """Plot one or more functions as 2D curves."""
    expressions = [expression] if isinstance(expression, str) else expression
    if not expressions:
        raise ValueError("expression must not be empty")
    parsed_expressions = [parse_expression(item) for item in expressions]
    variable_symbol = sympy.Symbol(variable)
    chosen_range = x_range or _auto_x_range(parsed_expressions)
    x_values = np.linspace(float(chosen_range[0]), float(chosen_range[1]), num_points)
    style = _plot_style(dark_mode)

    fig, ax = plt.subplots(figsize=tuple(figsize) if figsize else (8, 5), facecolor=style["facecolor"])
    ax.set_facecolor(style["axes_facecolor"])
    ax.tick_params(colors=style["text_color"])
    for spine in ax.spines.values():
        spine.set_color(style["text_color"])
    if grid:
        ax.grid(True, color=style["grid_color"], alpha=0.5)

    plotted_any = False
    for item, expr in zip(expressions, parsed_expressions):
        func = sympy.lambdify(variable_symbol, expr, modules=["numpy"])
        try:
            y_values = func(x_values)
        except Exception as error:
            raise ValueError(f"failed to evaluate '{item}': {error}") from error
        x_plot, y_plot = _sanitize_curve(x_values, y_values)
        if np.all(np.isnan(y_plot)):
            continue
        plotted_any = True
        ax.plot(x_plot, y_plot, label=item)

    if not plotted_any:
        raise ValueError("all plotted functions evaluated to NaN or infinity; try a different range")

    if len(expressions) > 1:
        ax.legend()
    ax.set_title(title or f"Plot of {', '.join(expressions)}", color=style["text_color"])
    ax.set_xlabel(xlabel or variable, color=style["text_color"])
    ax.set_ylabel(ylabel or "f(x)", color=style["text_color"])

    data_uri, _, _ = _figure_to_data_uri(fig)
    return {
        "result": f"Plot of {', '.join(expressions)} from {chosen_range[0]} to {chosen_range[1]}",
        "data_uri": data_uri,
        "x_range": [float(chosen_range[0]), float(chosen_range[1])],
        "latex": None,
    }


@tool_error_handler("plot_implicit")
async def plot_implicit(
    equation: str,
    x_range: list[float] | None = None,
    y_range: list[float] | None = None,
    grid_size: int = 200,
    title: str = "",
    dark_mode: bool = False,
) -> dict[str, Any]:
    """Plot implicit equations or inequalities in x and y."""
    parsed, relation = _parse_implicit_equation(equation)
    x_bounds = x_range or [-5.0, 5.0]
    y_bounds = y_range or [-5.0, 5.0]
    x_symbol = sympy.Symbol("x")
    y_symbol = sympy.Symbol("y")
    xx, yy = np.meshgrid(
        np.linspace(float(x_bounds[0]), float(x_bounds[1]), grid_size),
        np.linspace(float(y_bounds[0]), float(y_bounds[1]), grid_size),
    )
    style = _plot_style(dark_mode)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=style["facecolor"])
    ax.set_facecolor(style["axes_facecolor"])
    ax.tick_params(colors=style["text_color"])
    for spine in ax.spines.values():
        spine.set_color(style["text_color"])
    ax.grid(True, color=style["grid_color"], alpha=0.4)

    if isinstance(parsed, Relational):
        func = sympy.lambdify((x_symbol, y_symbol), parsed.lhs, modules=["numpy"])
        values = np.asarray(func(xx, yy), dtype=float)
        if relation in {">", ">="}:
            mask = values >= 0 if relation == ">=" else values > 0
        else:
            mask = values <= 0 if relation == "<=" else values < 0
        ax.contourf(xx, yy, mask.astype(float), levels=[0.5, 1.5], alpha=0.5, colors=["tab:blue"])
        ax.contour(xx, yy, values, levels=[0], colors=["white" if dark_mode else "black"], linewidths=1.5)
    else:
        func = sympy.lambdify((x_symbol, y_symbol), parsed, modules=["numpy"])
        values = np.asarray(func(xx, yy), dtype=float)
        contour = ax.contour(xx, yy, values, levels=[0], colors=["tab:blue"], linewidths=1.8)
        if len(contour.allsegs[0]) == 0:
            raise ValueError("implicit equation has no visible curve in the requested plotting window")

    ax.set_title(title or f"Implicit plot of {equation}", color=style["text_color"])
    ax.set_xlabel("x", color=style["text_color"])
    ax.set_ylabel("y", color=style["text_color"])
    data_uri, _, _ = _figure_to_data_uri(fig)
    return {
        "result": f"Implicit plot of {equation}",
        "data_uri": data_uri,
        "x_range": [float(x_bounds[0]), float(x_bounds[1])],
        "y_range": [float(y_bounds[0]), float(y_bounds[1])],
        "latex": None,
    }


def register(server: Any) -> None:
    """Register all rendering tools."""
    server.tool("to_latex")(to_latex)
    server.tool("render_math")(render_math)
    server.tool("plot_function")(plot_function)
    server.tool("plot_implicit")(plot_implicit)