"""Numerical computation tools using SciPy and NumPy."""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import integrate as scipy_integrate
from scipy import interpolate as scipy_interpolate
from scipy import optimize as scipy_optimize
import sympy

from ..utils.errors import tool_error_handler
from ..utils.latex_utils import safe_latex
from ..utils.parsing import parse_expression

_MAX_OUTPUT_POINTS = 500
_ODE_METHODS = {"RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"}


def _normalize_expression(expression: str) -> str:
    return expression.replace("abs(", "Abs(")


def _parse_numeric_expression(expression: str) -> sympy.Expr:
    return parse_expression(_normalize_expression(expression))


def _to_numpy_callable(expression: str, variable: str) -> tuple[sympy.Expr, sympy.Symbol, Callable[[Any], Any]]:
    variable_symbol = sympy.Symbol(variable.strip())
    expr = _parse_numeric_expression(expression)
    func = sympy.lambdify(variable_symbol, expr, "numpy")
    return expr, variable_symbol, func


def _top_level_split(expression: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(expression):
        if char in "([{" :
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            part = expression[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = expression[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _auto_break_points(
    expr: sympy.Expr,
    variable_symbol: sympy.Symbol,
    lower: float,
    upper: float,
    points: list[float] | None,
) -> list[float] | None:
    if points is not None:
        return sorted(set(points))
    if not np.isfinite(lower) or not np.isfinite(upper):
        return None
    break_points: set[float] = set()
    for atom in expr.atoms(sympy.Abs, sympy.sign):
        if not atom.args:
            continue
        try:
            roots = sympy.solve(atom.args[0], variable_symbol)
        except Exception:
            continue
        for root in roots:
            root_value = complex(sympy.N(root))
            if abs(root_value.imag) < 1e-12:
                point = float(root_value.real)
                if lower < point < upper:
                    break_points.add(point)
    return sorted(break_points) or None


def _quad_integrate(
    func: Callable[[Any], Any],
    lower: float,
    upper: float,
    *,
    points: list[float] | None,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[float, float, int, list[warnings.WarningMessage]]:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", scipy_integrate.IntegrationWarning)
        value, error, info = scipy_integrate.quad(
            func,
            lower,
            upper,
            points=points,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
            full_output=1,
        )[:3]
    return float(value), float(error), int(info.get("neval", 0)), caught_warnings


def _sampled_integral(
    func: Callable[[Any], Any],
    lower: float,
    upper: float,
    method: str,
) -> tuple[float, float, int]:
    x_values = np.linspace(lower, upper, 1000)
    y_values = np.asarray(func(x_values), dtype=float)
    if method == "simpson":
        value = scipy_integrate.simpson(y_values, x_values)
    else:
        value = np.trapezoid(y_values, x_values)
    return float(value), float("nan"), len(x_values)


def _root_sample_bracket(func: Callable[[Any], Any]) -> list[float] | None:
    grid = np.linspace(-10.0, 10.0, 401)
    values = np.asarray(func(grid), dtype=float)
    if np.allclose(values, values[0]):
        return None
    for left, right, left_value, right_value in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            continue
        if left_value == 0.0:
            return [float(left), float(left)]
        if left_value * right_value < 0:
            return [float(left), float(right)]
    return None


def _build_ode_locals(variable: str, y_variable: str) -> dict[str, Any]:
    return {
        variable: sympy.Symbol(variable),
        y_variable: sympy.IndexedBase(y_variable),
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "exp": sympy.exp,
        "sqrt": sympy.sqrt,
        "log": sympy.log,
        "Abs": sympy.Abs,
    }


def _curve_fit_callable(
    expr: sympy.Expr,
    x_symbol: sympy.Symbol,
    parameter_symbols: list[sympy.Symbol],
) -> Callable[..., Any]:
    func = sympy.lambdify((x_symbol, *parameter_symbols), expr, modules=["numpy"])

    def model(x_values: Any, *params: float) -> Any:
        return func(x_values, *params)

    return model


@tool_error_handler("numerical_integrate")
async def numerical_integrate(
    expression: str,
    variable: str = "x",
    lower: float = 0.0,
    upper: float = 1.0,
    method: str = "auto",
    points: list[float] | None = None,
    epsabs: float = 1.49e-8,
    epsrel: float = 1.49e-8,
    limit: int = 200,
) -> dict[str, Any]:
    """Numerically integrate a scalar function."""
    expr, variable_symbol, func = _to_numpy_callable(expression, variable)
    method_name = method.lower()
    if method_name not in {"auto", "quad", "romberg", "fixed_quad", "simpson", "trapezoid", "gaussian"}:
        raise ValueError("method must be one of: auto, quad, romberg, fixed_quad, simpson, trapezoid, gaussian")

    integration_points = _auto_break_points(expr, variable_symbol, lower, upper, points)

    if method_name in {"auto", "quad"}:
        value, error, neval, caught_warnings = _quad_integrate(
            func,
            lower,
            upper,
            points=integration_points,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )
        method_used = "quad"
        if method_name == "auto" and caught_warnings and np.isfinite(lower) and np.isfinite(upper):
            method_name = "romberg"
        else:
            return {
                "result": value,
                "error_estimate": error,
                "method_used": method_used,
                "neval": neval,
                "latex": f"I \\approx {value}",
            }

    if method_name == "romberg":
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("romberg requires finite integration limits")
        if hasattr(scipy_integrate, "romberg"):
            value = scipy_integrate.romberg(func, lower, upper, tol=epsabs, rtol=epsrel, divmax=20)
            error = float("nan")
            neval = 0
        else:
            value, error = scipy_integrate.quad(func, lower, upper, epsabs=epsabs, epsrel=epsrel)
            neval = 0
        return {
            "result": float(value),
            "error_estimate": float(error),
            "method_used": "romberg",
            "neval": neval,
            "latex": f"I \\approx {float(value)}",
        }

    if method_name in {"fixed_quad", "gaussian"}:
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("fixed_quad and gaussian require finite integration limits")
        value, _ = scipy_integrate.fixed_quad(func, lower, upper, n=50)
        return {
            "result": float(value),
            "error_estimate": float("nan"),
            "method_used": "fixed_quad" if method_name == "fixed_quad" else "gaussian",
            "neval": 50,
            "latex": f"I \\approx {float(value)}",
        }

    if method_name in {"simpson", "trapezoid"}:
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("simpson and trapezoid require finite integration limits")
        value, error, neval = _sampled_integral(func, lower, upper, method_name)
        return {
            "result": value,
            "error_estimate": error,
            "method_used": method_name,
            "neval": neval,
            "latex": f"I \\approx {value}",
        }

    raise ValueError("integration method did not produce a result")


@tool_error_handler("find_root")
async def find_root(
    expression: str,
    variable: str = "x",
    method: str = "auto",
    bracket: list[float] | None = None,
    guess: float | None = None,
    tol: float = 1e-12,
    maxiter: int = 1000,
) -> dict[str, Any]:
    """Find a root of a scalar function numerically."""
    expr, variable_symbol, func = _to_numpy_callable(expression, variable)
    derivative = sympy.lambdify(variable_symbol, sympy.diff(expr, variable_symbol), "numpy")
    method_name = method.lower()
    if method_name not in {"auto", "brentq", "newton", "secant", "bisection"}:
        raise ValueError("method must be one of: auto, brentq, newton, secant, bisection")

    sample_points = np.linspace(-5.0, 5.0, 21)
    sample_values = np.asarray(func(sample_points), dtype=float)
    if np.all(np.isfinite(sample_values)) and np.allclose(sample_values, sample_values[0]):
        return {
            "result": "function appears constant; no isolated root found",
            "f_at_root": float(sample_values[0]),
            "iterations": 0,
            "method_used": "none",
            "converged": False,
        }

    root_result: scipy_optimize.RootResults | None = None
    root_value: float | None = None
    method_used = method_name

    if bracket is not None:
        if len(bracket) != 2:
            raise ValueError("bracket must contain exactly two values")
        a, b = bracket
        if func(a) == func(b):
            raise ValueError("bracket endpoints must not evaluate to the same value")
        solver_method = "brentq" if method_name in {"auto", "brentq"} else "bisect"
        root_result = scipy_optimize.root_scalar(
            func,
            bracket=[a, b],
            method=solver_method,
            xtol=tol,
            maxiter=maxiter,
        )
        root_value = float(root_result.root)
        method_used = "brentq" if solver_method == "brentq" else "bisection"
    elif guess is not None:
        if method_name in {"auto", "newton"}:
            try:
                root_scalar = scipy_optimize.root_scalar(
                    func,
                    x0=guess,
                    fprime=derivative,
                    method="newton",
                    xtol=tol,
                    maxiter=maxiter,
                )
                root_result = root_scalar
                root_value = float(root_scalar.root)
                method_used = "newton"
            except Exception:
                root_scalar = scipy_optimize.root_scalar(
                    func,
                    x0=guess,
                    x1=guess + 1e-3 if guess == 0 else guess * 1.01,
                    method="secant",
                    xtol=tol,
                    maxiter=maxiter,
                )
                root_result = root_scalar
                root_value = float(root_scalar.root)
                method_used = "secant"
        elif method_name == "secant":
            root_result = scipy_optimize.root_scalar(
                func,
                x0=guess,
                x1=guess + 1e-3 if guess == 0 else guess * 1.01,
                method="secant",
                xtol=tol,
                maxiter=maxiter,
            )
            root_value = float(root_result.root)
            method_used = "secant"
        else:
            raise ValueError(f"method {method} requires a bracket")
    else:
        sampled_bracket = _root_sample_bracket(func)
        if sampled_bracket is None:
            raise ValueError("root search failed; provide a bracket or a good initial guess")
        if sampled_bracket[0] == sampled_bracket[1]:
            root_value = sampled_bracket[0]
            root_result = None
            method_used = "sampled"
        else:
            root_result = scipy_optimize.root_scalar(
                func,
                bracket=sampled_bracket,
                method="brentq",
                xtol=tol,
                maxiter=maxiter,
            )
            root_value = float(root_result.root)
            method_used = "brentq"

    if root_value is None:
        raise ValueError("root finding did not produce a result; provide a bracket or initial guess")

    f_at_root = float(func(root_value))
    return {
        "result": root_value,
        "f_at_root": f_at_root,
        "iterations": 0 if root_result is None else int(root_result.iterations),
        "method_used": method_used,
        "converged": abs(f_at_root) <= max(tol, 1e-10) if root_result is None else bool(root_result.converged),
        "note": "other roots may exist" if expr.is_polynomial() and sympy.degree(expr, variable_symbol) and sympy.degree(expr, variable_symbol) > 1 else None,
    }


@tool_error_handler("ode_solve")
async def ode_solve(
    dydt: str,
    y0: list[float],
    t_span: list[float],
    variable: str = "t",
    y_variable: str = "y",
    method: str = "RK45",
    max_step: float | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
    dense_output: bool = False,
    num_points: int = 100,
) -> dict[str, Any]:
    """Solve an initial value problem numerically."""
    if len(t_span) != 2:
        raise ValueError("t_span must contain exactly two values")
    if not y0:
        raise ValueError("y0 must contain at least one initial value")
    if method not in _ODE_METHODS:
        raise ValueError("method must be one of: RK45, RK23, DOP853, Radau, BDF, LSODA")

    t_symbol = sympy.Symbol(variable.strip())
    locals_dict = _build_ode_locals(t_symbol.name, y_variable.strip())
    rhs_parts = _top_level_split(dydt)

    if len(rhs_parts) == 1 and len(y0) == 1:
        y_symbol = sympy.Symbol(y_variable.strip())
        rhs_expr = sympy.sympify(_normalize_expression(rhs_parts[0]), locals={**locals_dict, y_variable.strip(): y_symbol})
        rhs_func = sympy.lambdify((t_symbol, y_symbol), rhs_expr, modules=["numpy"])

        def system_rhs(time: float, state: np.ndarray) -> np.ndarray:
            return np.asarray([rhs_func(time, state[0])], dtype=float)
    else:
        indexed_y = sympy.IndexedBase(y_variable.strip())
        rhs_exprs = [
            sympy.sympify(_normalize_expression(part), locals={**locals_dict, y_variable.strip(): indexed_y})
            for part in rhs_parts
        ]
        if len(rhs_exprs) != len(y0):
            raise ValueError("system dimension must match the number of initial conditions")
        rhs_funcs = [sympy.lambdify((t_symbol, indexed_y), expr, modules=["numpy"]) for expr in rhs_exprs]

        def system_rhs(time: float, state: np.ndarray) -> np.ndarray:
            return np.asarray([func(time, state) for func in rhs_funcs], dtype=float)

    capped_points = min(max(2, num_points), _MAX_OUTPUT_POINTS)
    t_eval = np.linspace(float(t_span[0]), float(t_span[1]), capped_points)
    solve_kwargs: dict[str, Any] = {
        "fun": system_rhs,
        "t_span": (float(t_span[0]), float(t_span[1])),
        "y0": np.asarray(y0, dtype=float),
        "method": method,
        "t_eval": t_eval,
        "rtol": rtol,
        "atol": atol,
        "dense_output": dense_output,
    }
    if max_step is not None:
        solve_kwargs["max_step"] = max_step

    solution = scipy_integrate.solve_ivp(**solve_kwargs)
    if not solution.success and method == "RK45":
        solve_kwargs["method"] = "BDF"
        solution = scipy_integrate.solve_ivp(**solve_kwargs)

    truncated_t = solution.t[:_MAX_OUTPUT_POINTS].tolist()
    truncated_y = solution.y[:, :_MAX_OUTPUT_POINTS].tolist()
    blew_up = not np.all(np.isfinite(solution.y))

    return {
        "result": f"ODE solved from t={float(t_span[0])} to t={float(t_span[1])} ({len(truncated_t)} steps)",
        "t_values": truncated_t,
        "y_values": truncated_y,
        "method_used": solve_kwargs["method"],
        "nfev": int(solution.nfev),
        "success": bool(solution.success and not blew_up),
        "message": "solution may blow up; returning partial results" if blew_up else solution.message,
    }


@tool_error_handler("interpolate")
async def interpolate_data(
    x_data: list[float],
    y_data: list[float],
    method: str = "cubic_spline",
    x_eval: list[float] | None = None,
    num_points: int = 100,
    k: int = 3,
) -> dict[str, Any]:
    """Interpolate scattered one-dimensional data."""
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    if len(x_data) < 2:
        raise ValueError("at least two data points are required")

    pairs = sorted(zip(x_data, y_data), key=lambda pair: pair[0])
    x_values = np.asarray([pair[0] for pair in pairs], dtype=float)
    y_values = np.asarray([pair[1] for pair in pairs], dtype=float)
    if len(np.unique(x_values)) != len(x_values):
        raise ValueError("duplicate x-values are not allowed for interpolation")

    method_name = method.lower()
    if x_eval is None:
        eval_points = np.linspace(float(x_values.min()), float(x_values.max()), min(max(2, num_points), _MAX_OUTPUT_POINTS))
    else:
        eval_points = np.asarray(x_eval, dtype=float)

    if method_name == "linear":
        interpolator = scipy_interpolate.interp1d(x_values, y_values, kind="linear", fill_value="extrapolate")
    elif method_name == "cubic_spline":
        if len(x_values) < 4:
            raise ValueError("cubic_spline requires at least four data points")
        interpolator = scipy_interpolate.CubicSpline(x_values, y_values)
    elif method_name == "akima":
        if len(x_values) < 5:
            raise ValueError("akima requires at least five data points")
        interpolator = scipy_interpolate.Akima1DInterpolator(x_values, y_values)
    elif method_name == "pchip":
        interpolator = scipy_interpolate.PchipInterpolator(x_values, y_values)
    elif method_name == "barycentric":
        interpolator = scipy_interpolate.BarycentricInterpolator(x_values, y_values)
    elif method_name == "krogh":
        interpolator = scipy_interpolate.KroghInterpolator(x_values, y_values)
    elif method_name == "polynomial":
        interpolator = scipy_interpolate.BarycentricInterpolator(x_values, y_values)
    else:
        raise ValueError("method must be one of: linear, cubic_spline, akima, pchip, barycentric, krogh, polynomial")

    y_eval = np.asarray(interpolator(eval_points), dtype=float)
    warning = None
    if np.any(eval_points < x_values.min()) or np.any(eval_points > x_values.max()):
        warning = "extrapolation was used outside the original data range"

    return {
        "result": f"Interpolated at {len(eval_points)} points",
        "x_eval": eval_points.tolist(),
        "y_eval": y_eval.tolist(),
        "method": method_name,
        "latex": None,
        "warning": warning,
        "spline_degree": k if method_name in {"cubic_spline", "linear"} else None,
    }


@tool_error_handler("curve_fit")
async def curve_fit(
    model: str,
    x_data: list[float],
    y_data: list[float],
    parameters: list[str] | None = None,
    p0: list[float] | None = None,
    sigma: list[float] | None = None,
    absolute_sigma: bool = False,
) -> dict[str, Any]:
    """Fit a nonlinear model to data with least squares."""
    if len(x_data) != len(y_data):
        raise ValueError("x_data and y_data must have the same length")
    if len(x_data) < 2:
        raise ValueError("at least two data points are required")

    x_symbol = sympy.Symbol("x")
    model_expr = _parse_numeric_expression(model)
    if parameters is None:
        parameter_symbols = sorted(
            [symbol for symbol in model_expr.free_symbols if symbol != x_symbol],
            key=lambda symbol: symbol.name,
        )
    else:
        parameter_symbols = [sympy.Symbol(name.strip()) for name in parameters if name.strip()]
    if not parameter_symbols:
        raise ValueError("no fit parameters were detected; provide parameter names explicitly")
    if len(parameter_symbols) >= len(x_data):
        raise ValueError("too many parameters for the number of data points")

    model_callable = _curve_fit_callable(model_expr, x_symbol, parameter_symbols)
    initial_guess = p0 if p0 is not None else [1.0] * len(parameter_symbols)
    sigma_values = None if sigma is None else np.asarray(sigma, dtype=float)
    if sigma_values is not None and len(sigma_values) != len(x_data):
        raise ValueError("sigma must have the same length as x_data")

    try:
        optimal_params, covariance = scipy_optimize.curve_fit(
            model_callable,
            np.asarray(x_data, dtype=float),
            np.asarray(y_data, dtype=float),
            p0=initial_guess,
            sigma=sigma_values,
            absolute_sigma=absolute_sigma,
            maxfev=10000,
        )
    except Exception as error:
        raise ValueError(f"curve fitting failed; try a better p0 guess: {error}") from error

    predictions = np.asarray(model_callable(np.asarray(x_data, dtype=float), *optimal_params), dtype=float)
    residuals = np.asarray(y_data, dtype=float) - predictions
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((np.asarray(y_data, dtype=float) - np.mean(y_data)) ** 2))
    r_squared = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    parameter_errors = np.sqrt(np.diag(covariance))

    parameter_map = {symbol.name: float(value) for symbol, value in zip(parameter_symbols, optimal_params)}
    error_map = {symbol.name: float(error) for symbol, error in zip(parameter_symbols, parameter_errors)}
    fitted_expr = sympy.simplify(model_expr.subs(parameter_map))
    result_parts = [
        f"{symbol.name} = {parameter_map[symbol.name]:.6g} ± {error_map[symbol.name]:.6g}"
        for symbol in parameter_symbols
    ]

    return {
        "result": ", ".join(result_parts),
        "parameters": parameter_map,
        "errors": error_map,
        "r_squared": float(r_squared),
        "covariance_matrix": covariance.tolist(),
        "latex": safe_latex(fitted_expr),
    }


def register(server: Any) -> None:
    """Register all numerical tools."""
    server.tool("numerical_integrate")(numerical_integrate)
    server.tool("find_root")(find_root)
    server.tool("ode_solve")(ode_solve)
    server.tool("interpolate")(interpolate_data)
    server.tool("curve_fit")(curve_fit)