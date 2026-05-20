"""Symbolic mathematics tools using SymPy."""

import asyncio
import re
from typing import Any

import mpmath
import sympy
from sympy.core.function import AppliedUndef
from sympy.core.relational import Equality, Relational
from sympy.integrals.manualintegrate import manualintegrate

from ..utils.parsing import parse_expression, expr_to_result_dict
from ..utils.errors import tool_error_handler

_SOLVE_TIMEOUT_SECONDS = 5.0
_DOMAIN_SETS: dict[str, sympy.Set] = {
    "real": sympy.S.Reals,
    "complex": sympy.S.Complexes,
    "integer": sympy.S.Integers,
    "positive": sympy.Interval.open(0, sympy.oo),
}


def _rebuild_from_factor_list(factors: tuple[sympy.Expr, list[tuple[sympy.Expr, int]]]) -> sympy.Expr:
    coefficient, factor_pairs = factors
    pieces: list[sympy.Expr] = []
    coefficient_expr = sympy.sympify(coefficient)
    if coefficient_expr != 1:
        pieces.append(coefficient_expr)
    for factor_expr, exponent in factor_pairs:
        if exponent == 1:
            pieces.append(factor_expr)
        else:
            pieces.append(sympy.Pow(factor_expr, exponent, evaluate=False))
    if not pieces:
        return sympy.Integer(1)
    if len(pieces) == 1:
        return pieces[0]
    return sympy.Mul(*pieces, evaluate=False)


def _integer_factorization(value: sympy.Expr) -> sympy.Expr:
    integer_value = int(value)
    if integer_value in {-1, 0, 1}:
        return sympy.Integer(integer_value)

    sign = -1 if integer_value < 0 else 1
    factors = sympy.factorint(abs(integer_value))
    pieces: list[sympy.Expr] = []
    if sign < 0:
        pieces.append(sympy.Integer(-1))
    for prime, exponent in factors.items():
        prime_expr = sympy.Integer(prime)
        if exponent == 1:
            pieces.append(prime_expr)
        else:
            pieces.append(sympy.Pow(prime_expr, exponent, evaluate=False))
    if len(pieces) == 1:
        return pieces[0]
    return sympy.Mul(*pieces, evaluate=False)


def _split_equation_parts(equation: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(equation):
        if char in "([{" :
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == "," and depth == 0:
            part = equation[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1
    tail = equation[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_relation(part: str) -> sympy.Basic:
    for operator in ("<=", ">=", "=", "<", ">"):
        if operator in part:
            left, right = part.split(operator, 1)
            left_expr = parse_expression(left)
            right_expr = parse_expression(right)
            if operator == "=":
                return sympy.Eq(left_expr, right_expr, evaluate=False)
            if operator == "<=":
                return sympy.Le(left_expr, right_expr, evaluate=False)
            if operator == ">=":
                return sympy.Ge(left_expr, right_expr, evaluate=False)
            if operator == "<":
                return sympy.Lt(left_expr, right_expr, evaluate=False)
            return sympy.Gt(left_expr, right_expr, evaluate=False)
    return parse_expression(part)


def _normalize_equation(item: sympy.Basic) -> sympy.Basic:
    if isinstance(item, Equality):
        return sympy.simplify(item.lhs - item.rhs)
    if isinstance(item, Relational):
        return item.func(item.lhs - item.rhs, 0)
    return sympy.simplify(item)


def _unsupported_functions(items: list[sympy.Basic]) -> list[str]:
    names: set[str] = set()
    for item in items:
        names.update(str(func.func) for func in item.atoms(AppliedUndef))
    return sorted(names)


def _detect_variables(items: list[sympy.Basic], variables: list[str] | None) -> list[sympy.Symbol]:
    if variables is not None:
        return [sympy.Symbol(variable.strip()) for variable in variables if variable.strip()]

    free_symbols: set[sympy.Symbol] = set()
    for item in items:
        free_symbols.update(item.free_symbols)
    return sorted(free_symbols, key=lambda symbol: symbol.name)


def _filter_solution(solution: Any, domain: str) -> bool:
    if domain == "complex":
        return True
    if domain == "real":
        return sympy.im(sympy.sympify(solution)).is_zero is not False
    if domain == "integer":
        return sympy.sympify(solution).is_Integer is True
    if domain == "positive":
        return sympy.sympify(solution).is_positive is True
    return True


def _filter_solution_list(solutions: list[Any], domain: str) -> list[Any]:
    return [solution for solution in solutions if _filter_solution(solution, domain)]


def _filter_solution_dicts(
    solutions: list[dict[sympy.Symbol, Any]],
    domain: str,
) -> list[dict[sympy.Symbol, Any]]:
    return [
        solution
        for solution in solutions
        if all(_filter_solution(value, domain) for value in solution.values())
    ]


def _build_steps(
    normalized_items: list[sympy.Basic],
    variables: list[sympy.Symbol],
) -> list[str]:
    if len(normalized_items) > 1:
        joined_variables = ", ".join(str(variable) for variable in variables)
        return [
            f"1. Interpreted the input as a system in {joined_variables}.",
            "2. Applied symbolic substitution/elimination to solve the system.",
        ]

    item = normalized_items[0]
    if isinstance(item, Relational) and not isinstance(item, Equality):
        return ["1. Rewrote the inequality with zero on one side.", "2. Reduced the inequality over the requested domain."]

    expr = sympy.sympify(item)
    if expr.is_polynomial() and variables:
        factored = sympy.factor(expr)
        if factored != expr:
            return [
                f"1. Factor: {factored} = 0",
                "2. Set each factor equal to zero and solve.",
            ]
    if variables and sympy.degree(expr, variables[0]) == 1:
        return [
            f"1. Rearranged the equation to {expr} = 0.",
            f"2. Isolated {variables[0]} and solved the linear equation.",
        ]
    return ["1. Rewrote the expression with zero on one side.", "2. Applied symbolic solving routines."]


def _format_single_variable_result(
    variable: sympy.Symbol,
    solutions: list[Any],
    domain: str,
) -> dict[str, Any]:
    if not solutions:
        return {
            "result": f"no solutions in the {domain} domain",
            "latex": None,
            "steps": None,
        }

    if len(solutions) == 1:
        solution_expr = sympy.sympify(solutions[0])
        return {
            "result": f"{variable} = {solution_expr}",
            "latex": sympy.latex(sympy.Eq(variable, solution_expr)),
            "steps": None,
        }

    finite_set = sympy.FiniteSet(*[sympy.sympify(solution) for solution in solutions])
    result = ", ".join(f"{variable} = {solution}" for solution in solutions)
    latex = " \\vee ".join(
        sympy.latex(sympy.Eq(variable, sympy.sympify(solution))) for solution in solutions
    )
    return {
        "result": result,
        "latex": latex or sympy.latex(finite_set),
        "steps": None,
    }


def _format_system_result(
    solutions: list[dict[sympy.Symbol, Any]],
    domain: str,
) -> dict[str, Any]:
    if not solutions:
        return {
            "result": f"no solutions in the {domain} domain",
            "latex": None,
            "steps": None,
        }

    first_solution = solutions[0]
    ordered_items = sorted(first_solution.items(), key=lambda item: item[0].name)
    result = ", ".join(f"{symbol} = {value}" for symbol, value in ordered_items)
    latex = sympy.latex(first_solution)
    return {
        "result": result,
        "latex": latex,
        "steps": None,
    }


def _format_set_result(
    variable: sympy.Symbol,
    solution_set: sympy.Set,
    domain: str,
) -> dict[str, Any]:
    if solution_set == sympy.EmptySet:
        return {
            "result": f"no solutions in the {domain} domain",
            "latex": sympy.latex(solution_set),
            "steps": None,
        }
    if solution_set in {sympy.S.Reals, sympy.S.Complexes, sympy.S.Integers}:
        return {
            "result": f"all {domain} numbers satisfy",
            "latex": sympy.latex(solution_set),
            "steps": None,
        }
    if isinstance(solution_set, sympy.ConditionSet):
        return {
            "result": f"infinitely many solutions: {solution_set}",
            "latex": sympy.latex(solution_set),
            "steps": None,
        }
    return {
        "result": f"{variable} in {solution_set}",
        "latex": sympy.latex(solution_set),
        "steps": None,
    }


def _solve_equations(
    equation: str,
    variables: list[str] | None,
    domain: str,
    show_steps: bool,
) -> dict[str, Any]:
    if domain not in _DOMAIN_SETS:
        raise ValueError("domain must be one of: real, complex, integer, positive")

    parts = _split_equation_parts(equation)
    if not parts:
        raise ValueError("equation must not be empty")

    parsed_items = [_parse_relation(part) for part in parts]
    unsupported_functions = _unsupported_functions(parsed_items)
    if unsupported_functions:
        raise ValueError(
            "Could not solve: the expression contains an unsupported special function: "
            + ", ".join(unsupported_functions)
        )

    normalized_items = [_normalize_equation(item) for item in parsed_items]
    variable_symbols = _detect_variables(parsed_items, variables)
    steps = _build_steps(normalized_items, variable_symbols) if show_steps else None

    if not variable_symbols:
        if all(sympy.sympify(item) == 0 for item in normalized_items):
            return {
                "result": f"all {domain} numbers satisfy",
                "latex": sympy.latex(_DOMAIN_SETS[domain]),
                "steps": steps,
            }
        return {
            "result": f"no solutions in the {domain} domain",
            "latex": sympy.latex(sympy.EmptySet),
            "steps": steps,
        }

    domain_set = _DOMAIN_SETS[domain]

    if len(parsed_items) == 1 and isinstance(parsed_items[0], Relational) and not isinstance(parsed_items[0], Equality):
        if len(variable_symbols) != 1:
            raise ValueError("inequalities currently support exactly one variable")
        variable = variable_symbols[0]
        inequality = normalized_items[0]
        try:
            solution_set = sympy.solve_univariate_inequality(inequality, variable, relational=False)
        except NotImplementedError:
            solution_set = sympy.reduce_inequalities([inequality], [variable])
        if isinstance(solution_set, sympy.Set):
            solution_set = solution_set.intersect(domain_set)
            result = _format_set_result(variable, solution_set, domain)
        else:
            result = {
                "result": f"infinitely many solutions: {solution_set}",
                "latex": sympy.latex(solution_set),
                "steps": None,
            }
        result["steps"] = steps
        return result

    if len(normalized_items) == 1 and len(variable_symbols) == 1:
        variable = variable_symbols[0]
        expr = sympy.sympify(normalized_items[0])
        if expr == 0:
            return {
                "result": f"all {domain} numbers satisfy",
                "latex": sympy.latex(domain_set),
                "steps": steps,
            }
        if expr.is_number:
            return {
                "result": f"no solutions in the {domain} domain",
                "latex": sympy.latex(sympy.EmptySet),
                "steps": steps,
            }

        use_solveset = expr.has(sympy.sin, sympy.cos, sympy.tan, sympy.cot, sympy.sec, sympy.csc)
        raw_solutions: list[Any] = []
        if not use_solveset:
            raw_solutions = sympy.solve(expr, variable)
        if raw_solutions:
            filtered_solutions = _filter_solution_list(raw_solutions, domain)
            result = _format_single_variable_result(variable, filtered_solutions, domain)
            result["steps"] = steps
            return result

        solution_set = sympy.solveset(expr, variable, domain=domain_set)
        if isinstance(solution_set, sympy.FiniteSet):
            filtered_solutions = _filter_solution_list(list(solution_set), domain)
            result = _format_single_variable_result(variable, filtered_solutions, domain)
            result["steps"] = steps
            return result
        if isinstance(solution_set, sympy.ConditionSet):
            return {
                "result": "no closed-form symbolic solution exists; try a numerical approach",
                "latex": sympy.latex(solution_set),
                "steps": steps,
            }
        result = _format_set_result(variable, solution_set, domain)
        result["steps"] = steps
        return result

    equation_list = [sympy.sympify(item) for item in normalized_items]
    solution_dicts = sympy.solve(equation_list, variable_symbols, dict=True)
    filtered_solution_dicts = _filter_solution_dicts(solution_dicts, domain)
    if filtered_solution_dicts:
        result = _format_system_result(filtered_solution_dicts, domain)
        result["steps"] = steps
        return result

    return {
        "result": f"no solutions in the {domain} domain",
        "latex": sympy.latex(sympy.EmptySet),
        "steps": steps,
    }


def _format_special_value(value: sympy.Expr) -> str:
    if value == sympy.oo:
        return "∞"
    if value == -sympy.oo:
        return "-∞"
    if value == sympy.zoo:
        return "complex infinity"
    return str(value)


def _parse_point(point: str) -> sympy.Expr:
    normalized = point.strip().replace("infinity", "oo")
    return parse_expression(normalized)


def _parse_integration_limits(limits: list[str] | None) -> tuple[sympy.Expr, sympy.Expr] | None:
    if limits is None:
        return None
    if len(limits) != 2:
        raise ValueError("limits must contain exactly two entries: [lower, upper]")
    return (_parse_point(limits[0]), _parse_point(limits[1]))


def _build_numeric_approximation(
    expr: sympy.Expr,
    variable: sympy.Symbol,
    parsed_limits: tuple[sympy.Expr, sympy.Expr],
) -> float | None:
    try:
        approximation = sympy.N(sympy.Integral(expr, (variable, *parsed_limits)))
        if approximation.is_real:
            return float(approximation)
    except Exception:
        pass

    try:
        function = sympy.lambdify(variable, expr, modules=["mpmath"])
        approximation = mpmath.quad(function, [parsed_limits[0], parsed_limits[1]])
        return float(approximation)
    except Exception:
        return None


def _integration_steps(
    expr: sympy.Expr,
    variable: sympy.Symbol,
    result: sympy.Expr,
    parsed_limits: tuple[sympy.Expr, sympy.Expr] | None,
) -> list[str]:
    steps: list[str] = []
    try:
        manual_result = manualintegrate(expr, variable)
        if manual_result != result:
            steps.append(f"1. Manual integration candidate: {manual_result}")
        else:
            steps.append(f"1. Manual integration: {manual_result}")
    except Exception:
        steps.append("1. Attempted manual integration steps, but no step-by-step form was available.")

    if parsed_limits is not None:
        steps.append(
            f"2. Evaluated the antiderivative between {parsed_limits[0]} and {parsed_limits[1]}."
        )
    else:
        steps.append("2. Returned the symbolic antiderivative.")
    return steps


def _series_steps(point_expr: sympy.Expr) -> list[str] | None:
    if point_expr.is_finite is False:
        return ["1. Expanded the function asymptotically at infinity."]
    return None


def _build_sympify_locals(
    extra_symbols: list[str] | dict[str, sympy.Symbol] | None = None,
    extra_functions: list[str] | None = None,
) -> dict[str, Any]:
    locals_dict: dict[str, Any] = {
        "Eq": sympy.Eq,
        "Derivative": sympy.Derivative,
        "diff": sympy.diff,
        "Function": sympy.Function,
        "Heaviside": sympy.Heaviside,
        "DiracDelta": sympy.DiracDelta,
        "exp": sympy.exp,
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "log": sympy.log,
        "sqrt": sympy.sqrt,
        "pi": sympy.pi,
        "E": sympy.E,
        "oo": sympy.oo,
        "Integral": sympy.Integral,
        "Sum": sympy.Sum,
        "Product": sympy.Product,
    }
    if isinstance(extra_symbols, dict):
        locals_dict.update(extra_symbols)
    else:
        for symbol_name in extra_symbols or []:
            if symbol_name:
                locals_dict[symbol_name] = sympy.Symbol(symbol_name)
    for function_name in extra_functions or []:
        if function_name:
            locals_dict[function_name] = sympy.Function(function_name)
    return locals_dict


def _function_from_string(function: str) -> tuple[sympy.Expr, sympy.Symbol, str]:
    match = re.fullmatch(r"\s*([A-Za-z_]\w*)\(([^()]+)\)\s*", function)
    if not match:
        raise ValueError("function must look like y(x) or f(t)")
    function_name, variable_name = match.groups()
    variable_symbol = sympy.Symbol(variable_name.strip())
    function_expr = sympy.Function(function_name)(variable_symbol)
    return function_expr, variable_symbol, function_name


def _parse_functional_expression(
    expression: str,
    *,
    extra_symbols: list[str] | dict[str, sympy.Symbol] | None = None,
    extra_functions: list[str] | None = None,
) -> sympy.Basic:
    locals_dict = _build_sympify_locals(extra_symbols, extra_functions)
    try:
        return sympy.sympify(expression.strip().replace("^", "**"), locals=locals_dict)
    except sympy.SympifyError as error:
        raise ValueError(f"Could not parse symbolic expression: {error}") from error


def _parse_equation_or_expression(
    equation: str,
    *,
    extra_symbols: list[str] | dict[str, sympy.Symbol] | None = None,
    extra_functions: list[str] | None = None,
) -> sympy.Basic:
    if "=" in equation and "Eq(" not in equation:
        left, right = equation.split("=", 1)
        left_expr = _parse_functional_expression(
            left,
            extra_symbols=extra_symbols,
            extra_functions=extra_functions,
        )
        right_expr = _parse_functional_expression(
            right,
            extra_symbols=extra_symbols,
            extra_functions=extra_functions,
        )
        return sympy.Eq(left_expr, right_expr, evaluate=False)
    return _parse_functional_expression(
        equation,
        extra_symbols=extra_symbols,
        extra_functions=extra_functions,
    )


def _parse_ics(
    ics: dict[str, str] | None,
    function_expr: sympy.Expr,
    variable_symbol: sympy.Symbol,
    function_name: str,
) -> dict[sympy.Expr, sympy.Expr] | None:
    if not ics:
        return None

    parsed: dict[sympy.Expr, sympy.Expr] = {}
    for key, value in ics.items():
        value_expr = _parse_functional_expression(value, extra_symbols=[variable_symbol.name], extra_functions=[function_name])
        stripped_key = key.strip()
        if stripped_key.isdigit():
            order = int(stripped_key)
            lhs = function_expr if order == 0 else sympy.diff(function_expr, variable_symbol, order)
            parsed[lhs.subs(variable_symbol, 0)] = value_expr
            continue
        normalized_key = stripped_key.replace("'", "Derivative")
        if stripped_key.startswith(f"{function_name}(") or stripped_key.startswith("Derivative("):
            lhs_expr = _parse_functional_expression(
                stripped_key,
                extra_symbols=[variable_symbol.name],
                extra_functions=[function_name],
            )
            parsed[lhs_expr] = value_expr
            continue
        raise ValueError("ics keys must be derivative orders like '0' or expressions like y(0)")
    return parsed


@tool_error_handler("simplify")
async def simplify(expression: str) -> dict[str, Any]:
    """
    Simplify a mathematical expression algebraically.

    Args:
        expression: A SymPy-compatible expression string.
                    Examples: "(x**2 - 1)/(x - 1)", "sin(x)**2 + cos(x)**2"

    Returns:
        Standard result dictionary containing the simplified expression.
    """
    expr = parse_expression(expression)
    result = sympy.simplify(expr)
    if expr.is_number:
        result = sympy.sympify(result)
    return expr_to_result_dict(result)


@tool_error_handler("expand")
async def expand(expression: str) -> dict[str, Any]:
    """
    Expand a mathematical expression.

    Args:
        expression: SymPy-compatible expression.
                    Examples: "(x+1)**3", "(a+b)*(c+d)"

    Returns:
        Standard result dictionary containing the expanded expression.
    """
    expr = parse_expression(expression)
    result = sympy.expand(expr)
    if expr.is_polynomial():
        result = sympy.expand(expr, mul=True, multinomial=True)
    return expr_to_result_dict(result)


@tool_error_handler("factor")
async def factor(expression: str) -> dict[str, Any]:
    """
    Factor a mathematical expression over the rational numbers.

    Args:
        expression: SymPy-compatible polynomial or rational expression.
                    Examples: "x**3 - 6*x**2 + 11*x - 6", "x**4 - 1"

    Returns:
        Standard result dictionary containing the factored expression.
    """
    expr = parse_expression(expression)
    if expr.is_number:
        return expr_to_result_dict(_integer_factorization(expr))

    try:
        result = sympy.factor(expr)
        if result == expr:
            result = sympy.factor(expr, modulus=None)
        if result == expr:
            factor_list_result = sympy.factor_list(expr)
            result = _rebuild_from_factor_list(factor_list_result)
    except sympy.PolynomialError as error:
        raise ValueError(
            f"Could not factor expression as a polynomial over the rationals: {error}"
        ) from error

    return expr_to_result_dict(result)


@tool_error_handler("solve")
async def solve(
    equation: str,
    variables: list[str] | None = None,
    domain: str = "real",
    show_steps: bool = False,
) -> dict[str, Any]:
    """
    Solve equations or systems of equations symbolically.

    Args:
        equation: Equation string or comma-separated system.
        variables: Optional variable names to solve for.
        domain: "real", "complex", "integer", or "positive".
        show_steps: Whether to include a short human-readable explanation.
    """
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_solve_equations, equation, variables, domain, show_steps),
            timeout=_SOLVE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return {
            "result": "solving timed out after 5 seconds; try a numerical approach",
            "latex": None,
            "steps": [
                "1. Parsing succeeded.",
                "2. Symbolic solving exceeded the 5 second timeout.",
            ] if show_steps else None,
        }


@tool_error_handler("symbolic_integrate")
async def symbolic_integrate(
    expression: str,
    variable: str = "x",
    limits: list[str] | None = None,
    show_steps: bool = False,
    method: str = "auto",
) -> dict[str, Any]:
    """Compute symbolic indefinite or definite integrals."""
    expr = parse_expression(expression)
    variable_symbol = sympy.Symbol(variable.strip())
    parsed_limits = _parse_integration_limits(limits)

    if method not in {"auto", "risch", "manual", "meijerg"}:
        raise ValueError("method must be one of: auto, risch, manual, meijerg")

    if parsed_limits is None:
        integral_target: sympy.Expr | sympy.Integral = sympy.Integral(expr, variable_symbol)
    else:
        integral_target = sympy.Integral(expr, (variable_symbol, *parsed_limits))

    if method == "manual":
        result = manualintegrate(expr, variable_symbol)
        if parsed_limits is not None:
            antiderivative = result
            result = sympy.simplify(
                antiderivative.subs(variable_symbol, parsed_limits[1])
                - antiderivative.subs(variable_symbol, parsed_limits[0])
            )
    elif method == "risch":
        result = sympy.integrate(expr, variable_symbol, risch=True)
        if parsed_limits is not None:
            result = sympy.simplify(
                result.subs(variable_symbol, parsed_limits[1])
                - result.subs(variable_symbol, parsed_limits[0])
            )
    else:
        result = sympy.integrate(expr, (variable_symbol, *parsed_limits)) if parsed_limits else sympy.integrate(expr, variable_symbol)
        if result.has(sympy.Integral):
            result = integral_target.doit()
        if result.has(sympy.Integral) or method == "meijerg":
            result = sympy.integrate(
                expr,
                (variable_symbol, *parsed_limits) if parsed_limits else variable_symbol,
                meijerg=True,
            )

    response = expr_to_result_dict(result)
    if show_steps:
        response["steps"] = _integration_steps(expr, variable_symbol, result, parsed_limits)

    if result in {sympy.oo, -sympy.oo, sympy.zoo}:
        response["result"] = f"diverges to {_format_special_value(result)}"
        response["latex"] = sympy.latex(result)
        return response

    if result.has(sympy.Integral):
        response["result"] = str(result)
        response["steps"] = (response.get("steps") or []) + [
            "3. No closed-form antiderivative was found; returning an unevaluated integral."
        ]

    if parsed_limits is not None:
        approximation = None
        if result.has(sympy.Integral) or not result.is_number:
            approximation = _build_numeric_approximation(expr, variable_symbol, parsed_limits)
        else:
            try:
                approximation = float(sympy.N(result))
            except Exception:
                approximation = None
        if approximation is not None:
            response["numeric_approximation"] = approximation

    return response


@tool_error_handler("symbolic_diff")
async def symbolic_diff(
    expression: str,
    variable: str = "x",
    order: int = 1,
    show_steps: bool = False,
) -> dict[str, Any]:
    """Compute symbolic derivatives of arbitrary order."""
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")

    expr = parse_expression(expression)
    variable_symbol = sympy.Symbol(variable.strip())

    steps: list[str] | None = None
    if show_steps and order > 1:
        steps = []
        current = expr
        for step_index in range(1, order + 1):
            current = sympy.diff(current, variable_symbol)
            steps.append(f"{step_index}. Derivative order {step_index}: {current}")
        result = current
    else:
        result = sympy.diff(expr, variable_symbol, order)

    result = sympy.simplify(result)
    return expr_to_result_dict(result, steps=steps)


@tool_error_handler("limit")
async def limit(
    expression: str,
    variable: str = "x",
    point: str = "0",
    direction: str = "+-",
) -> dict[str, Any]:
    """Compute limits of expressions."""
    if direction not in {"+", "-", "+-"}:
        raise ValueError("direction must be one of: +, -, +-")

    expr = parse_expression(expression)
    variable_symbol = sympy.Symbol(variable.strip())
    point_expr = _parse_point(point)

    if direction == "+-":
        result = sympy.Limit(expr, variable_symbol, point_expr, dir="+-").doit()
        if isinstance(result, sympy.Limit) or result == sympy.zoo:
            left_result = sympy.Limit(expr, variable_symbol, point_expr, dir="-").doit()
            right_result = sympy.Limit(expr, variable_symbol, point_expr, dir="+").doit()
            if left_result != right_result:
                return {
                    "result": "does not exist (two-sided)",
                    "latex": None,
                    "steps": None,
                    "left_limit": _format_special_value(left_result),
                    "right_limit": _format_special_value(right_result),
                }
            result = left_result
    else:
        result = sympy.Limit(expr, variable_symbol, point_expr, dir=direction).doit()

    return {
        "result": _format_special_value(result),
        "latex": sympy.latex(result),
        "steps": None,
    }


@tool_error_handler("series")
async def series(
    expression: str,
    variable: str = "x",
    point: str = "0",
    order: int = 6,
) -> dict[str, Any]:
    """Compute Taylor or Laurent series expansions."""
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")

    expr = parse_expression(expression)
    variable_symbol = sympy.Symbol(variable.strip())
    point_expr = _parse_point(point)

    try:
        series_result = sympy.series(expr, variable_symbol, point_expr, order)
    except Exception as error:
        raise ValueError(
            f"Could not compute series expansion; try a different point or lower order: {error}"
        ) from error

    polynomial_part = series_result.removeO() if hasattr(series_result, "removeO") else series_result
    response = {
        "result": str(series_result),
        "latex": sympy.latex(series_result),
        "polynomial_part": str(polynomial_part),
        "steps": _series_steps(point_expr),
    }
    if point_expr.is_finite and expr.has(sympy.log, sympy.Abs):
        response["steps"] = (response.get("steps") or []) + [
            "2. The expansion may include branch-sensitive terms near the expansion point."
        ]
    if point_expr.is_finite and expr.subs(variable_symbol, point_expr) in {sympy.zoo, sympy.oo, -sympy.oo, sympy.nan}:
        response["steps"] = (response.get("steps") or []) + [
            "3. The expansion point is a singularity, so this is a Laurent series."
        ]
    return response


@tool_error_handler("symbolic_sum")
async def symbolic_sum(
    expression: str,
    index: str = "n",
    start: str = "0",
    end: str = "oo",
) -> dict[str, Any]:
    """Compute symbolic finite or infinite summations."""
    index_symbol = sympy.Symbol(index.strip())
    expr = parse_expression(expression)
    start_expr = _parse_point(start)
    end_expr = _parse_point(end)

    summation_result = sympy.summation(expr, (index_symbol, start_expr, end_expr))
    result = sympy.simplify(summation_result)
    if result.has(sympy.Sum):
        unevaluated_sum = sympy.Sum(expr, (index_symbol, start_expr, end_expr))
        return {
            "result": str(unevaluated_sum),
            "latex": sympy.latex(unevaluated_sum),
            "steps": ["No closed form was found; returning an unevaluated symbolic sum."],
        }
    if end_expr == sympy.oo:
        convergence = sympy.simplify(sympy.limit(expr, index_symbol, sympy.oo))
        if convergence not in {0, sympy.Integer(0)} and not result.is_finite:
            return {
                "result": f"diverges to {_format_special_value(result)}",
                "latex": sympy.latex(result),
                "steps": None,
            }
    return expr_to_result_dict(result)


@tool_error_handler("symbolic_product")
async def symbolic_product(
    expression: str,
    index: str = "n",
    start: str = "1",
    end: str = "oo",
) -> dict[str, Any]:
    """Compute symbolic finite or infinite products."""
    index_symbol = sympy.Symbol(index.strip())
    expr = parse_expression(expression)
    start_expr = _parse_point(start)
    end_expr = _parse_point(end)

    product_result = sympy.product(expr, (index_symbol, start_expr, end_expr))
    result = sympy.simplify(product_result)
    if result.has(sympy.Product):
        unevaluated_product = sympy.Product(expr, (index_symbol, start_expr, end_expr))
        return {
            "result": str(unevaluated_product),
            "latex": sympy.latex(unevaluated_product),
            "steps": ["No closed form was found; returning an unevaluated symbolic product."],
        }
    if end_expr == sympy.oo and result in {sympy.oo, -sympy.oo, sympy.zoo}:
        return {
            "result": f"diverges to {_format_special_value(result)}",
            "latex": sympy.latex(result),
            "steps": None,
        }
    return expr_to_result_dict(result)


@tool_error_handler("dsolve")
async def dsolve(
    equation: str,
    function: str = "y(x)",
    ics: dict[str, str] | None = None,
    hint: str = "default",
) -> dict[str, Any]:
    """Solve ordinary differential equations symbolically."""
    function_expr, variable_symbol, function_name = _function_from_string(function)
    equation_parts = _split_equation_parts(equation)
    parsed_equations = [
        _parse_equation_or_expression(
            part,
            extra_symbols=[variable_symbol.name],
            extra_functions=[function_name],
        )
        for part in equation_parts
    ]
    normalized_equations = [
        parsed if isinstance(parsed, Equality) else sympy.Eq(parsed, 0, evaluate=False)
        for parsed in parsed_equations
    ]
    parsed_ics = _parse_ics(ics, function_expr, variable_symbol, function_name)

    classification: tuple[str, ...] = tuple()
    if len(normalized_equations) == 1:
        try:
            classification = tuple(sympy.classify_ode(normalized_equations[0], function_expr))
        except Exception:
            classification = tuple()

    solve_hint = "default" if hint == "default" else hint
    try:
        result = sympy.dsolve(
            normalized_equations if len(normalized_equations) > 1 else normalized_equations[0],
            function_expr if len(normalized_equations) == 1 else None,
            hint=solve_hint,
            ics=parsed_ics,
        )
    except NotImplementedError:
        return {
            "result": "no closed-form symbolic solution exists; try a numerical approach",
            "latex": None,
            "steps": None,
            "classification": list(classification),
        }

    if isinstance(result, list):
        result_expr = result[0]
        result_text = "; ".join(str(item) for item in result)
        latex = sympy.latex(result)
    else:
        result_expr = result
        result_text = str(result)
        latex = sympy.latex(result)
    if isinstance(result_expr, Equality):
        result_text = f"{result_expr.lhs} = {result_expr.rhs}"

    return {
        "result": result_text,
        "latex": latex,
        "steps": None,
        "classification": list(classification),
    }


@tool_error_handler("laplace_transform")
async def laplace_transform(
    expression: str,
    t_var: str = "t",
    s_var: str = "s",
    noconds: bool = True,
) -> dict[str, Any]:
    """Compute the Laplace transform of a time-domain expression."""
    t_symbol = sympy.Symbol(t_var.strip(), real=True)
    s_symbol = sympy.Symbol(s_var.strip())
    expr = _parse_functional_expression(
        expression,
        extra_symbols={t_symbol.name: t_symbol, s_symbol.name: s_symbol},
    )

    try:
        result = sympy.laplace_transform(expr, t_symbol, s_symbol, noconds=noconds)
    except Exception as error:
        raise ValueError(f"Could not compute Laplace transform: {error}") from error

    transform_expr = result if noconds else result[0]
    response = {
        "result": f"F({s_symbol}) = {transform_expr}",
        "latex": sympy.latex(transform_expr),
        "steps": None,
    }
    if not noconds:
        response["conditions"] = [str(item) for item in result[1:]]
    return response


@tool_error_handler("inverse_laplace")
async def inverse_laplace(
    expression: str,
    s_var: str = "s",
    t_var: str = "t",
) -> dict[str, Any]:
    """Compute the inverse Laplace transform."""
    s_symbol = sympy.Symbol(s_var.strip())
    t_symbol = sympy.Symbol(t_var.strip(), real=True)
    expr = _parse_functional_expression(
        expression,
        extra_symbols={s_symbol.name: s_symbol, t_symbol.name: t_symbol},
    )

    try:
        result = sympy.inverse_laplace_transform(expr, s_symbol, t_symbol)
    except Exception as error:
        raise ValueError(f"Could not compute inverse Laplace transform: {error}") from error

    return {
        "result": f"f({t_symbol}) = {result}",
        "latex": sympy.latex(result),
        "steps": None,
    }


def register(server: Any) -> None:
    """Register all symbolic tools with the MCP server."""
    server.tool("simplify")(simplify)
    server.tool("expand")(expand)
    server.tool("factor")(factor)
    server.tool("solve")(solve)
    server.tool("symbolic_integrate")(symbolic_integrate)
    server.tool("symbolic_diff")(symbolic_diff)
    server.tool("limit")(limit)
    server.tool("series")(series)
    server.tool("symbolic_sum")(symbolic_sum)
    server.tool("symbolic_product")(symbolic_product)
    server.tool("dsolve")(dsolve)
    server.tool("laplace_transform")(laplace_transform)
    server.tool("inverse_laplace")(inverse_laplace)