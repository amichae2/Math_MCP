import pytest

from math_mcp.tools.symbolic import (
    dsolve,
    expand,
    factor,
    inverse_laplace,
    laplace_transform,
    limit,
    series,
    simplify,
    solve,
    symbolic_diff,
    symbolic_integrate,
    symbolic_product,
    symbolic_sum,
)


@pytest.mark.asyncio
async def test_simplify_polynomial(sample_expressions) -> None:
    result = await simplify(expression="(x+1)*(x-1)")

    assert result["result"] == "x**2 - 1"
    assert result.get("error") is None


@pytest.mark.asyncio
async def test_simplify_trig_identity(sample_expressions) -> None:
    result = await simplify(expression=sample_expressions["trig"])

    assert result["result"] == "1"
    assert result.get("error") is None


@pytest.mark.asyncio
async def test_simplify_rational_expression(sample_expressions) -> None:
    result = await simplify(expression=sample_expressions["rational"])

    assert result["result"] == "x + 1"


@pytest.mark.asyncio
async def test_simplify_constant_folding() -> None:
    result = await simplify(expression="2+2")

    assert result["result"] == "4"


@pytest.mark.asyncio
async def test_simplify_invalid_input() -> None:
    result = await simplify(expression="")

    assert result["error"] is not None


@pytest.mark.asyncio
async def test_expand_binomial_theorem() -> None:
    result = await expand(expression="(x+1)**3")

    assert result["result"] == "x**3 + 3*x**2 + 3*x + 1"


@pytest.mark.asyncio
async def test_expand_polynomial_multiplication() -> None:
    result = await expand(expression="(a+b)*(c+d)")

    for term in ["a*c", "a*d", "b*c", "b*d"]:
        assert term in result["result"]


@pytest.mark.asyncio
async def test_expand_trig_like_square() -> None:
    result = await expand(expression="(sin(x)+cos(x))**2")

    assert "sin(x)**2" in result["result"]
    assert "cos(x)**2" in result["result"]


@pytest.mark.asyncio
async def test_factor_polynomial() -> None:
    result = await factor(expression="x**3 - 6*x**2 + 11*x - 6")

    assert all(piece in result["result"] for piece in ["x - 3", "x - 2", "x - 1"])


@pytest.mark.asyncio
async def test_factor_irreducible_polynomial() -> None:
    result = await factor(expression="x**2 + 1")

    assert result["result"] == "x**2 + 1"


@pytest.mark.asyncio
async def test_factor_integer() -> None:
    result = await factor(expression="12")

    assert result["result"] == "2**2*3"


@pytest.mark.asyncio
async def test_factor_multivariate() -> None:
    result = await factor(expression="x**2*y - x*y")

    assert "x*y" in result["result"]
    assert "x - 1" in result["result"]


@pytest.mark.asyncio
async def test_solve_linear() -> None:
    result = await solve(equation="2*x + 3 = 7", variables=["x"])

    assert result["result"] == "x = 2"


@pytest.mark.asyncio
async def test_solve_quadratic() -> None:
    result = await solve(equation="x**2 - 4 = 0", variables=["x"])

    assert "-2" in result["result"] and "2" in result["result"]


@pytest.mark.asyncio
async def test_solve_cubic() -> None:
    result = await solve(equation="x**3 - 6*x**2 + 11*x - 6 = 0", variables=["x"])

    for root in ["1", "2", "3"]:
        assert root in result["result"]


@pytest.mark.asyncio
async def test_solve_system_of_two_equations() -> None:
    result = await solve(equation="x + y = 3, x - y = 1", show_steps=True)

    assert result["result"] == "x = 2, y = 1"
    assert result["steps"] is not None


@pytest.mark.asyncio
async def test_solve_inequality() -> None:
    result = await solve(equation="x**2 - 5*x + 6 > 0", variables=["x"])

    assert "Interval" in result["result"] or "Union" in result["result"]


@pytest.mark.asyncio
async def test_solve_no_solution_real_domain() -> None:
    result = await solve(equation="x**2 + 1 = 0", variables=["x"], domain="real")

    assert result["result"] == "no solutions in the real domain"


@pytest.mark.asyncio
async def test_solve_infinite_solutions() -> None:
    result = await solve(equation="x = x")

    assert result["result"] == "all real numbers satisfy"


@pytest.mark.asyncio
async def test_solve_trig_periodic_solution() -> None:
    result = await solve(equation="sin(x) = 0", variables=["x"], domain="real")

    assert "ImageSet" in result["result"] or "Integers" in result["result"] or "pi" in result["result"]


@pytest.mark.asyncio
async def test_symbolic_integrate_polynomial() -> None:
    result = await symbolic_integrate(expression="x**2", variable="x")

    assert result["result"] == "x**3/3"


@pytest.mark.asyncio
async def test_symbolic_integrate_trig() -> None:
    result = await symbolic_integrate(expression="sin(x)", variable="x")

    assert result["result"] == "-cos(x)"


@pytest.mark.asyncio
async def test_symbolic_integrate_exponential() -> None:
    result = await symbolic_integrate(expression="exp(x)", variable="x")

    assert result["result"] == "exp(x)"


@pytest.mark.asyncio
async def test_symbolic_integrate_definite() -> None:
    result = await symbolic_integrate(expression="x**2", variable="x", limits=["0", "1"])

    assert result["result"] == "1/3"
    assert result["numeric_approximation"] == pytest.approx(1 / 3)


@pytest.mark.asyncio
async def test_symbolic_integrate_improper() -> None:
    result = await symbolic_integrate(expression="exp(-x)", variable="x", limits=["0", "oo"])

    assert result["result"] == "1"


@pytest.mark.asyncio
async def test_symbolic_integrate_special_function_non_elementary() -> None:
    result = await symbolic_integrate(expression="exp(-x**2)", variable="x")

    assert "erf" in result["result"] or "Integral" in result["result"]


@pytest.mark.asyncio
async def test_symbolic_diff_polynomial() -> None:
    result = await symbolic_diff(expression="x**3")

    assert result["result"] == "3*x**2"


@pytest.mark.asyncio
async def test_symbolic_diff_trig() -> None:
    result = await symbolic_diff(expression="sin(x)")

    assert result["result"] == "cos(x)"


@pytest.mark.asyncio
async def test_symbolic_diff_product_rule() -> None:
    result = await symbolic_diff(expression="x*exp(x)")

    assert result["result"] == "(x + 1)*exp(x)"


@pytest.mark.asyncio
async def test_symbolic_diff_chain_rule() -> None:
    result = await symbolic_diff(expression="sin(x**2)")

    assert result["result"] == "2*x*cos(x**2)"


@pytest.mark.asyncio
async def test_symbolic_diff_higher_order() -> None:
    result = await symbolic_diff(expression="x**5", order=3, show_steps=True)

    assert result["result"] == "60*x**2"
    assert len(result["steps"]) == 3


@pytest.mark.asyncio
async def test_symbolic_diff_absolute_value() -> None:
    result = await symbolic_diff(expression="Abs(x)")

    assert "sign" in result["result"] or "Piecewise" in result["result"]


@pytest.mark.asyncio
async def test_limit_finite() -> None:
    result = await limit(expression="sin(x)/x", point="0")

    assert result["result"] == "1"


@pytest.mark.asyncio
async def test_limit_infinite() -> None:
    result = await limit(expression="(1 + 1/x)**x", point="oo")

    assert result["result"] in {"E", "2.71828182845905"}


@pytest.mark.asyncio
async def test_limit_one_sided() -> None:
    result = await limit(expression="1/x", point="0", direction="+")

    assert result["result"] == "∞"


@pytest.mark.asyncio
async def test_limit_two_sided_equal() -> None:
    result = await limit(expression="(x**2 - 1)/(x - 1)", point="1")

    assert result["result"] == "2"


@pytest.mark.asyncio
async def test_limit_two_sided_unequal() -> None:
    result = await limit(expression="1/x", point="0", direction="+-")

    assert result["result"] == "does not exist (two-sided)"
    assert result["left_limit"] == "-∞"
    assert result["right_limit"] == "∞"


@pytest.mark.asyncio
async def test_limit_lhopital_case() -> None:
    result = await limit(expression="(exp(x) - 1)/x", point="0")

    assert result["result"] == "1"


@pytest.mark.asyncio
async def test_series_sin_around_zero() -> None:
    result = await series(expression="sin(x)", order=6)

    assert result["result"] == "x - x**3/6 + x**5/120 + O(x**6)"


@pytest.mark.asyncio
async def test_series_cos_around_zero() -> None:
    result = await series(expression="cos(x)", order=6)

    assert result["polynomial_part"] == "x**4/24 - x**2/2 + 1"


@pytest.mark.asyncio
async def test_series_exp_around_zero() -> None:
    result = await series(expression="exp(x)", order=5)

    assert "O(x**5)" in result["result"]


@pytest.mark.asyncio
async def test_series_log_around_zero() -> None:
    result = await series(expression="log(1 + x)", order=5)

    assert "x" in result["polynomial_part"]
    assert "x**2" in result["polynomial_part"]


@pytest.mark.asyncio
async def test_series_nonzero_point() -> None:
    result = await series(expression="exp(x)", point="1", order=4)

    assert "O((x - 1)**4" in result["result"] or "O(x - 1" in result["result"]


@pytest.mark.asyncio
async def test_symbolic_sum_basel() -> None:
    """sum(1/n**2, n=1..oo) = pi**2/6."""
    result = await symbolic_sum(expression="1/n**2", index="n", start="1", end="oo")
    assert "pi" in result["result"].lower() or "6" in result["result"]


@pytest.mark.asyncio
async def test_symbolic_sum_geometric() -> None:
    """sum(x**n, n=0..oo) = 1/(1-x) for |x|<1."""
    result = await symbolic_sum(expression="x**n", index="n", start="0", end="oo")
    assert result["result"] is not None


@pytest.mark.asyncio
async def test_symbolic_product_factorial() -> None:
    """product(n, n=1..5) = 120."""
    result = await symbolic_product(expression="n", index="n", start="1", end="5")
    assert "120" in result["result"]


@pytest.mark.asyncio
async def test_dsolve_exponential() -> None:
    """y' = y gives y = C1*exp(x)."""
    result = await dsolve(equation="Derivative(y(x), x) - y(x) = 0", function="y(x)")
    assert "exp" in result["result"] or "C1" in result["result"]


@pytest.mark.asyncio
async def test_dsolve_with_ics() -> None:
    """y' = y with y(0) = 1 gives exp(x)."""
    result = await dsolve(
        equation="Derivative(y(x), x) - y(x) = 0",
        function="y(x)",
        ics={"0": "1"},
    )
    assert "exp" in result["result"]


@pytest.mark.asyncio
async def test_laplace_exp() -> None:
    """L{exp(-a*t)} = 1/(s+a)."""
    result = await laplace_transform(expression="exp(-a*t)", t_var="t", s_var="s")
    assert "1/(a + s)" in result["result"] or "1/(s + a)" in result["result"]


@pytest.mark.asyncio
async def test_inverse_laplace() -> None:
    """L^-1{1/(s**2 + 1)} = sin(t)."""
    result = await inverse_laplace(expression="1/(s**2 + 1)", s_var="s", t_var="t")
    assert "sin" in result["result"]
