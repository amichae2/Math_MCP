"""Expression parsing utilities using SymPy."""

from typing import Any

import sympy

_SAFE_LOCALS: dict[str, Any] = {
    "sin": sympy.sin,
    "cos": sympy.cos,
    "tan": sympy.tan,
    "cot": sympy.cot,
    "sec": sympy.sec,
    "csc": sympy.csc,
    "asin": sympy.asin,
    "acos": sympy.acos,
    "atan": sympy.atan,
    "atan2": sympy.atan2,
    "sinh": sympy.sinh,
    "cosh": sympy.cosh,
    "tanh": sympy.tanh,
    "asinh": sympy.asinh,
    "acosh": sympy.acosh,
    "atanh": sympy.atanh,
    "log": sympy.log,
    "ln": sympy.log,
    "exp": sympy.exp,
    "sqrt": sympy.sqrt,
    "cbrt": sympy.cbrt,
    "Abs": sympy.Abs,
    "sign": sympy.sign,
    "floor": sympy.floor,
    "ceiling": sympy.ceiling,
    "factorial": sympy.factorial,
    "binomial": sympy.binomial,
    "gamma": sympy.gamma,
    "erf": sympy.erf,
    "erfc": sympy.erfc,
    "Piecewise": sympy.Piecewise,
    "pi": sympy.pi,
    "E": sympy.E,
    "I": sympy.I,
    "oo": sympy.oo,
    "zoo": sympy.zoo,
    "nan": sympy.nan,
    "e": sympy.E,
    "i": sympy.I,
    "inf": sympy.oo,
    "infinity": sympy.oo,
}

_MAX_EXPRESSION_LENGTH = 10_000


class ExpressionParseError(ValueError):
    """Raised when an expression string cannot be parsed."""


def parse_expression(
    expr_str: str,
    *,
    max_length: int = _MAX_EXPRESSION_LENGTH,
) -> sympy.Expr:
    """
    Safely parse a string into a SymPy expression.

    Args:
        expr_str: The expression string (e.g., "x**2 + sin(x)")
        max_length: Maximum allowed string length (default 10KB)

    Returns:
        A sympy.Expr object.

    Raises:
        ExpressionParseError: If the string is empty, too long, or unparseable.
    """
    if not expr_str or not expr_str.strip():
        raise ExpressionParseError("Expression string is empty")

    if len(expr_str) > max_length:
        raise ExpressionParseError(
            f"Expression too long ({len(expr_str)} chars, max {max_length})"
        )

    normalized = expr_str.strip().replace("^", "**")

    try:
        expr = sympy.sympify(normalized, locals=_SAFE_LOCALS)
    except sympy.SympifyError as error:
        raise ExpressionParseError(f"Cannot parse expression: {error}") from error
    except SyntaxError as error:
        raise ExpressionParseError(f"Syntax error in expression: {error}") from error

    return expr


def parse_variables(var_strs: list[str]) -> list[sympy.Symbol]:
    """Convert a list of variable name strings to SymPy Symbols."""
    return [sympy.Symbol(value.strip()) for value in var_strs if value.strip()]


def expr_to_result_dict(
    expr: Any,
    *,
    latex: str | None = None,
    steps: list[str] | None = None,
) -> dict[str, Any]:
    """Convert a SymPy expression/result into the standard result dict."""
    try:
        result_str = str(expr)
    except Exception:
        result_str = repr(expr)

    try:
        latex_str = latex or sympy.latex(expr)
    except Exception:
        latex_str = None

    return {
        "result": result_str,
        "latex": latex_str,
        "steps": steps or [],
    }