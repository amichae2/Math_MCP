"""Discrete math and number theory tools using SymPy and mpmath."""

import asyncio
import json
import math
from functools import reduce
from typing import Any

import mpmath
import sympy
from sympy import ntheory
from sympy.ntheory import factor_ as factor_algorithms
from sympy.ntheory.modular import crt
from sympy.ntheory.residue_ntheory import discrete_log as sympy_discrete_log
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.utilities.iterables import partitions as sympy_partitions

from ..utils.parsing import expr_to_result_dict, parse_expression
from ..utils.errors import tool_error_handler
from ..utils.latex_utils import safe_latex

_PRIMALITY_TIMEOUT_SECONDS = 30.0


def _parse_integer(value: str | int, *, name: str = "value") -> int:
    if isinstance(value, int):
        return value
    expr = parse_expression(str(value))
    if expr.free_symbols:
        raise ValueError(f"{name} must evaluate to an integer without free symbols")
    if expr.is_integer is False:
        raise ValueError(f"{name} must evaluate to an integer")
    numeric = sympy.Integer(expr)
    return int(numeric)


def _format_factorization(n: int, factors: dict[int, int]) -> tuple[str, list[list[int]], str]:
    ordered = sorted(factors.items(), key=lambda item: abs(item[0]))
    pieces = []
    factor_pairs: list[list[int]] = []
    for prime, exponent in ordered:
        factor_pairs.append([int(prime), int(exponent)])
        if exponent == 1:
            pieces.append(f"{prime}")
        else:
            pieces.append(f"{prime}^{exponent}")
    result_text = f"{n} = " + " × ".join(pieces)
    factor_expr = sympy.Integer(1)
    for prime, exponent in ordered:
        factor_expr *= sympy.Integer(prime) ** exponent
    return result_text, factor_pairs, sympy.latex(sympy.Eq(sympy.Integer(n), factor_expr, evaluate=False))


def _trial_factorization(n: int) -> tuple[dict[int, int], int]:
    remaining = abs(n)
    factors: dict[int, int] = {}
    if remaining in {0, 1}:
        return ({remaining: 1} if remaining else {0: 1}, 1)
    divisor = 2
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            remaining //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    return factors, remaining


def _pollard_factorization(n: int, *, method: str) -> tuple[dict[int, int], bool, str]:
    remaining = abs(n)
    factors: dict[int, int] = {}
    method_used = method
    while remaining > 1 and not sympy.isprime(remaining):
        if method == "pollard_pm1":
            factor = factor_algorithms.pollard_pm1(remaining)
        else:
            factor = factor_algorithms.pollard_rho(remaining)
        if factor in {None, 1, remaining}:
            break
        multiplicity = 0
        while remaining % factor == 0:
            multiplicity += 1
            remaining //= factor
        factors[int(factor)] = factors.get(int(factor), 0) + multiplicity
    if remaining > 1:
        if sympy.isprime(remaining):
            factors[int(remaining)] = factors.get(int(remaining), 0) + 1
            remaining = 1
        else:
            method_used += " (partial)"
    return factors, remaining == 1, method_used


def _parse_number_list(values: list[str] | list[int]) -> list[int]:
    return [_parse_integer(value, name="number") for value in values]


def _euclidean_steps(a: int, b: int) -> list[str]:
    steps: list[str] = []
    x, y = abs(a), abs(b)
    while y != 0:
        q, r = divmod(x, y)
        steps.append(f"{x} = {q}×{y} + {r}")
        x, y = y, r
    return steps


def _parse_crt_values(value: str | None, *, name: str) -> list[int]:
    if value is None:
        raise ValueError(f"{name} is required for crt")
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [_parse_integer(item, name=name) for item in parsed]
    except Exception:
        pass
    return [_parse_integer(item.strip(), name=name) for item in value.split(",") if item.strip()]


def _format_large_integer(value: int | sympy.Expr) -> str:
    integer = int(sympy.Integer(value))
    if abs(integer) > 10**15:
        return f"{integer:.6e}"
    return f"{integer:,}"


def _partition_dict_to_list(partition_dict: dict[int, int]) -> list[int]:
    values: list[int] = []
    for part, multiplicity in sorted(partition_dict.items(), reverse=True):
        values.extend([int(part)] * int(multiplicity))
    return values


def _count_partitions_exact(n: int, k: int) -> int:
    dp = [[0] * (k + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    for part in range(1, n + 1):
        for total in range(part, n + 1):
            for pieces in range(1, k + 1):
                dp[total][pieces] += dp[total - part][pieces - 1]
    return dp[n][k]


def _enumerate_small_solutions(expr: sympy.Expr, variables: list[sympy.Symbol], bound: int) -> list[list[int]]:
    solutions: list[list[int]] = []
    ranges = [range(-bound, bound + 1) for _ in variables]
    for values in __import__("itertools").product(*ranges):
        substitution = dict(zip(variables, values))
        if sympy.simplify(expr.subs(substitution)) == 0:
            solutions.append([int(value) for value in values])
    return solutions


def _parse_equation_to_expr(equation: str) -> sympy.Expr:
    if "=" in equation:
        left, right = equation.split("=", 1)
        return sympy.simplify(parse_expression(left) - parse_expression(right))
    return sympy.simplify(parse_expression(equation))


@tool_error_handler("factor_int")
async def factor_int(
    n: str,
    method: str = "auto",
) -> dict[str, Any]:
    """Factor an integer into prime factors."""
    integer = _parse_integer(n, name="n")
    if integer == 0:
        return {"result": "0 has no finite prime factorization", "factors": [], "is_complete": False, "method_used": "none", "latex": None, "steps": None}
    if abs(integer) == 1:
        return {"result": f"{integer} has no prime factors", "factors": [], "is_complete": True, "method_used": "none", "latex": None, "steps": None}

    warning = None
    if len(str(abs(integer))) > 80:
        warning = "factorization may be very slow for numbers larger than 10^80"

    method_name = method.lower()
    if method_name not in {"auto", "trial", "pollard_rho", "pollard_pm1"}:
        raise ValueError("method must be one of: auto, trial, pollard_rho, pollard_pm1")

    if method_name == "auto":
        factors = sympy.factorint(integer)
        method_used = "factorint"
        is_complete = all(sympy.isprime(abs(prime)) for prime in factors)
    elif method_name == "trial":
        factors, remaining = _trial_factorization(integer)
        is_complete = remaining == 1 or sympy.isprime(remaining)
        if remaining not in {0, 1}:
            factors[int(remaining)] = factors.get(int(remaining), 0) + 1
        if integer < 0:
            factors[-1] = factors.get(-1, 0) + 1
        method_used = "trial"
    else:
        factors, is_complete, method_used = _pollard_factorization(integer, method=method_name)
        if integer < 0:
            factors[-1] = factors.get(-1, 0) + 1

    result_text, factor_pairs, latex = _format_factorization(integer, factors)
    response = {
        "result": result_text,
        "factors": factor_pairs,
        "is_complete": is_complete,
        "method_used": method_used,
        "latex": latex,
        "steps": None,
    }
    if warning is not None:
        response["warning"] = warning
    return response


def _rigorous_isprime(integer: int) -> dict[str, Any]:
    primality = sympy.isprime(integer)
    return {
        "result": f"{integer} is {'prime' if primality else 'composite'}",
        "is_prime": bool(primality),
        "method": "deterministic isprime",
        "latex": None,
        "steps": None,
    }


@tool_error_handler("is_prime")
async def is_prime(
    n: str,
    rigorous: bool = True,
) -> dict[str, Any]:
    """Test whether an integer is prime."""
    integer = _parse_integer(n, name="n")
    if rigorous:
        try:
            return await asyncio.wait_for(asyncio.to_thread(_rigorous_isprime, integer), timeout=_PRIMALITY_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            return {
                "result": "rigorous primality proving timed out after 30 seconds",
                "is_prime": None,
                "method": "timeout",
                "latex": None,
                "steps": None,
            }
    primality = ntheory.primetest.mr(abs(integer), [2, 3, 5, 7, 11, 13]) if abs(integer) > 1 else False
    return {
        "result": f"{integer} is {'probably prime' if primality else 'composite'}",
        "is_prime": bool(primality),
        "method": "Miller-Rabin",
        "latex": None,
        "steps": None,
    }


@tool_error_handler("primes_range")
async def primes_range(
    start: int,
    end: int,
    limit: int = 1000,
) -> dict[str, Any]:
    """List primes in a range or generate the first N primes."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if end < start:
        count = min(start, limit)
        primes = [int(sympy.prime(index)) for index in range(1, count + 1)]
        return {
            "result": f"Generated first {len(primes)} primes",
            "primes": primes,
            "count": len(primes),
            "prime_density": None,
            "latex": None,
            "steps": None,
        }

    primes = [int(prime) for _, prime in zip(range(limit), sympy.primerange(start, end + 1))]
    range_size = max(end - start + 1, 1)
    return {
        "result": f"{len(primes)} primes between {start} and {end}",
        "primes": primes,
        "count": len(primes),
        "prime_density": float(len(primes) / range_size),
        "latex": None,
        "steps": None,
    }


@tool_error_handler("gcd_lcm")
async def gcd_lcm(
    numbers: list[str] | list[int],
    operation: str = "gcd",
    extended: bool = False,
) -> dict[str, Any]:
    """Compute GCD, LCM, or Euclidean algorithm steps."""
    parsed = _parse_number_list(numbers)
    if not parsed:
        raise ValueError("numbers must not be empty")
    op = operation.lower()
    if op not in {"gcd", "lcm", "euclidean_steps"}:
        raise ValueError("operation must be one of: gcd, lcm, euclidean_steps")

    if op == "gcd":
        value = reduce(math.gcd, parsed)
        response = {
            "result": f"gcd({', '.join(str(number) for number in parsed)}) = {value}",
            "value": int(value),
            "latex": None,
            "steps": None,
        }
        if extended:
            if len(parsed) != 2:
                raise ValueError("extended gcd is only supported for exactly two numbers")
            x_coeff, y_coeff, gcd_value = sympy.gcdex(parsed[0], parsed[1])
            response["bezout"] = {
                "x": int(x_coeff),
                "y": int(y_coeff),
                "equation": f"{parsed[0]}×{int(x_coeff)} + {parsed[1]}×{int(y_coeff)} = {int(gcd_value)}",
            }
        return response

    if op == "lcm":
        value = reduce(math.lcm, parsed)
        return {
            "result": f"lcm({', '.join(str(number) for number in parsed)}) = {value}",
            "value": int(value),
            "latex": None,
            "steps": None,
        }

    if len(parsed) != 2:
        raise ValueError("euclidean_steps requires exactly two numbers")
    steps = _euclidean_steps(parsed[0], parsed[1])
    return {
        "result": f"Euclidean algorithm for {parsed[0]} and {parsed[1]}",
        "steps": steps,
        "latex": None,
    }


@tool_error_handler("modular")
async def modular(
    operation: str,
    a: str,
    b: str | None = None,
    mod: str | None = None,
    power: str | None = None,
) -> dict[str, Any]:
    """Perform modular arithmetic operations."""
    op = operation.lower()
    if op == "crt":
        remainders = _parse_crt_values(a, name="remainders")
        moduli = _parse_crt_values(mod, name="moduli")
        if len(remainders) != len(moduli):
            raise ValueError("remainders and moduli must have the same length")
        value, modulus = crt(moduli, remainders)
        return {
            "result": f"x ≡ {int(value)} mod {int(modulus)}",
            "value": int(value),
            "modulus": int(modulus),
            "latex": None,
            "steps": None,
        }

    a_value = _parse_integer(a, name="a")
    modulus = _parse_integer(mod, name="mod") if mod is not None else None
    if op not in {"add", "subtract", "multiply", "inverse", "power", "sqrt", "discrete_log"}:
        raise ValueError("operation must be one of: add, subtract, multiply, inverse, power, sqrt, discrete_log, crt")
    if modulus is None or modulus <= 0:
        raise ValueError("mod must be a positive integer")

    if op == "add":
        b_value = _parse_integer(b, name="b")
        value = (a_value + b_value) % modulus
        return {"result": f"({a_value} + {b_value}) mod {modulus} = {value}", "value": value, "latex": None, "steps": None}
    if op == "subtract":
        b_value = _parse_integer(b, name="b")
        value = (a_value - b_value) % modulus
        return {"result": f"({a_value} - {b_value}) mod {modulus} = {value}", "value": value, "latex": None, "steps": None}
    if op == "multiply":
        b_value = _parse_integer(b, name="b")
        value = (a_value * b_value) % modulus
        return {"result": f"({a_value} × {b_value}) mod {modulus} = {value}", "value": value, "latex": None, "steps": None}
    if op == "inverse":
        try:
            value = int(sympy.mod_inverse(a_value, modulus))
        except ValueError as error:
            raise ValueError("modular inverse does not exist for these values") from error
        return {
            "result": f"{a_value}^-1 mod {modulus} = {value}",
            "value": value,
            "verification": f"{a_value}×{value} mod {modulus} = {(a_value * value) % modulus}",
            "latex": None,
            "steps": None,
        }
    if op == "power":
        exponent = _parse_integer(power, name="power")
        value = pow(a_value, exponent, modulus)
        return {"result": f"{a_value}^{exponent} mod {modulus} = {value}", "value": value, "latex": None, "steps": None}
    if op == "sqrt":
        roots = sqrt_mod(a_value, modulus, all_roots=True)
        if roots is None:
            raise ValueError("no modular square root exists")
        root_values = [int(root) for root in roots]
        return {"result": f"sqrt({a_value}) mod {modulus} = {root_values}", "value": root_values, "latex": None, "steps": None}
    value = int(sympy_discrete_log(modulus, a_value, _parse_integer(b, name="b")))
    return {"result": f"log_{b}({a_value}) mod {modulus} = {value}", "value": value, "latex": None, "steps": None}


@tool_error_handler("combinatorics")
async def combinatorics(
    operation: str,
    n: int,
    k: int | None = None,
    as_formula: bool = False,
) -> dict[str, Any]:
    """Evaluate combinatorial number sequences and counting functions."""
    op = operation.lower()
    value: Any
    formula: str | None = None
    if op == "permutations":
        if k is None:
            raise ValueError("k is required for permutations")
        value = math.perm(n, k)
        formula = f"{n}!/({n-k})!"
    elif op == "combinations":
        if k is None:
            raise ValueError("k is required for combinations")
        value = math.comb(n, k)
        formula = f"{n}!/({k}!×{n-k}!)"
    elif op == "combinations_with_replacement":
        if k is None:
            raise ValueError("k is required for combinations_with_replacement")
        value = math.comb(n + k - 1, k)
        formula = f"({n}+{k}-1)!/({k}!×({n}-1)!)"
    elif op == "factorial":
        value = math.factorial(n)
        formula = f"{n}!"
    elif op == "double_factorial":
        value = int(sympy.factorial2(n))
        formula = f"{n}!!"
    elif op == "subfactorial":
        value = int(sympy.subfactorial(n))
        formula = f"!{n}"
    elif op == "catalan":
        value = sympy.catalan(n)
        formula = f"Catalan({n})"
    elif op == "bell":
        value = sympy.bell(n)
        formula = f"Bell({n})"
    elif op == "fibonacci":
        value = sympy.fibonacci(n)
        formula = f"F_{n}"
    elif op == "bernoulli":
        value = sympy.bernoulli(n)
        formula = f"B_{n}"
    elif op == "euler":
        value = sympy.euler(n)
        formula = f"E_{n}"
    elif op == "stirling1":
        if k is None:
            raise ValueError("k is required for stirling1")
        value = sympy.functions.combinatorial.numbers.stirling(n, k, kind=1)
        formula = f"s({n}, {k})"
    elif op == "stirling2":
        if k is None:
            raise ValueError("k is required for stirling2")
        value = sympy.functions.combinatorial.numbers.stirling(n, k, kind=2)
        formula = f"S({n}, {k})"
    elif op == "multinomial":
        if k is None:
            raise ValueError("k is required for multinomial and is interpreted as the number of equal buckets")
        if n % k != 0:
            raise ValueError("for this multinomial helper, n must be divisible by k")
        coeffs = [n // k] * k
        value = sympy.multinomial_coefficients(len(coeffs), tuple(coeffs))[tuple(coeffs)]
        formula = f"{n}!/" + "×".join(f"{coefficient}!" for coefficient in coeffs)
    else:
        raise ValueError("unsupported combinatorics operation")

    numeric_value = sympy.N(value)
    formatted_value = _format_large_integer(value) if getattr(value, "is_integer", False) else str(value)
    response = {
        "result": f"{operation} = {formatted_value}",
        "value": int(value) if getattr(value, "is_integer", False) else str(value),
        "latex": safe_latex(value),
        "steps": None,
    }
    if as_formula:
        response["formula"] = formula
    if abs(float(numeric_value)) > 1e15:
        response["scientific_notation"] = f"{float(numeric_value):.6e}"
    return response


@tool_error_handler("partitions")
async def partitions(
    n: int,
    operation: str = "count",
    k: int | None = None,
) -> dict[str, Any]:
    """Count or list integer partitions."""
    if n < 0:
        raise ValueError("n must be non-negative")
    op = operation.lower()
    if op not in {"count", "list", "count_parts", "list_parts"}:
        raise ValueError("operation must be one of: count, list, count_parts, list_parts")

    if op == "count":
        value = int(sympy.functions.combinatorial.numbers.partition(n))
        return {"result": f"p({n}) = {value}", "value": value, "latex": safe_latex(sympy.Integer(value)), "steps": None}

    if op == "count_parts":
        if k is None:
            raise ValueError("k is required for count_parts")
        value = _count_partitions_exact(n, k)
        return {"result": f"partitions of {n} into exactly {k} parts = {value}", "value": value, "latex": None, "steps": None}

    total_count = int(sympy.functions.combinatorial.numbers.partition(n))
    if n > 50:
        return {
            "result": f"listing partitions for n > 50 is disabled; p({n}) = {total_count}",
            "value": total_count,
            "latex": None,
            "steps": None,
            "warning": "returning only the count because the list grows too quickly",
        }

    listed = [_partition_dict_to_list(item) for item in sympy_partitions(n, size=False)]
    if op == "list_parts":
        if k is None:
            raise ValueError("k is required for list_parts")
        listed = [item for item in listed if len(item) == k]
    return {
        "result": f"{len(listed)} partitions of {n}",
        "partitions": listed,
        "count": len(listed),
        "latex": None,
        "steps": None,
    }


@tool_error_handler("diophantine")
async def diophantine(
    equation: str,
    method: str = "auto",
    bound: int = 10,
) -> dict[str, Any]:
    """Solve Diophantine equations over the integers."""
    method_name = method.lower()
    equations = [item.strip() for item in equation.split(",") if item.strip()]
    if not equations:
        raise ValueError("equation must not be empty")

    if len(equations) == 1:
        expr = _parse_equation_to_expr(equations[0])
        variables = sorted(expr.free_symbols, key=lambda symbol: symbol.name)
        if method_name in {"auto", "linear"} and len(variables) == 2 and sympy.Poly(expr, *variables).total_degree() == 1:
            a_coeff = int(sympy.diff(expr, variables[0]))
            b_coeff = int(sympy.diff(expr, variables[1]))
            c_coeff = int(-expr.subs({variables[0]: 0, variables[1]: 0}))
            gcd_value = math.gcd(a_coeff, b_coeff)
            if c_coeff % gcd_value != 0:
                return {"result": "no integer solutions", "general_solution": None, "small_solutions": [], "latex": None, "steps": None}
            x_coeff, y_coeff, _ = sympy.gcdex(a_coeff, b_coeff)
            x0 = int(x_coeff * (c_coeff // gcd_value))
            y0 = int(y_coeff * (c_coeff // gcd_value))
            parameter = sympy.Symbol("t", integer=True)
            general_solution = (
                f"{variables[0]} = {x0} + {b_coeff // gcd_value}*t, "
                f"{variables[1]} = {y0} - {a_coeff // gcd_value}*t"
            )
            small_solutions = _enumerate_small_solutions(expr, variables, bound)
            return {
                "result": general_solution,
                "general_solution": general_solution,
                "parametric_variable": str(parameter),
                "small_solutions": small_solutions,
                "latex": None,
                "steps": None,
            }

        solution_set = sympy.diophantine(expr)
        if not solution_set:
            return {"result": "no integer solutions", "general_solution": None, "small_solutions": [], "latex": None, "steps": None}
        small_solutions = _enumerate_small_solutions(expr, variables, bound) if variables else []
        first_solution = next(iter(solution_set))
        parameters = sorted({str(symbol) for solution in solution_set for symbol in getattr(solution, "free_symbols", set()) if symbol not in variables})
        return {
            "result": str(first_solution),
            "general_solution": str(first_solution),
            "parametric_variable": parameters[0] if parameters else None,
            "small_solutions": small_solutions,
            "latex": safe_latex(first_solution),
            "steps": None,
        }

    parsed_equations = [_parse_equation_to_expr(item) for item in equations]
    variables = list(
        dict.fromkeys(
            symbol
            for expr in parsed_equations
            for symbol in sorted(expr.free_symbols, key=lambda item: item.name)
        )
    )
    solutions = sympy.linsolve(parsed_equations, *variables)
    if not solutions:
        return {"result": "no integer solutions", "general_solution": None, "small_solutions": [], "latex": None, "steps": None}
    first_solution = next(iter(solutions))
    small_solutions = []
    if all(solution.free_symbols <= set(variables) for solution in solutions):
        small_solutions = [[int(value) for value in first_solution]]
    return {
        "result": str(first_solution),
        "general_solution": str(first_solution),
        "parametric_variable": None,
        "small_solutions": small_solutions,
        "latex": safe_latex(first_solution),
        "steps": None,
    }


def register(server: Any) -> None:
    """Register all discrete math tools."""
    server.tool("factor_int")(factor_int)
    server.tool("is_prime")(is_prime)
    server.tool("primes_range")(primes_range)
    server.tool("gcd_lcm")(gcd_lcm)
    server.tool("modular")(modular)
    server.tool("combinatorics")(combinatorics)
    server.tool("partitions")(partitions)
    server.tool("diophantine")(diophantine)