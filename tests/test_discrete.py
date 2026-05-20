import pytest

from math_mcp.tools.discrete import (
    combinatorics,
    diophantine,
    factor_int,
    gcd_lcm,
    is_prime,
    modular,
    partitions,
    primes_range,
)


@pytest.mark.asyncio
async def test_factor_int_returns_prime_factorization() -> None:
    result = await factor_int("123456789")

    assert result["is_complete"] is True
    assert result["factors"] == [[3, 2], [3607, 1], [3803, 1]]


@pytest.mark.asyncio
async def test_factor_int_product_matches_original() -> None:
    result = await factor_int("123456789")

    product = 1
    for prime, exponent in result["factors"]:
        product *= prime**exponent
    assert product == 123456789


@pytest.mark.asyncio
async def test_is_prime_identifies_prime() -> None:
    result = await is_prime("1234567891")

    assert result["is_prime"] is True


@pytest.mark.asyncio
async def test_is_prime_identifies_composite() -> None:
    result = await is_prime("1234567890")

    assert result["is_prime"] is False


@pytest.mark.asyncio
async def test_primes_range_lists_expected_primes() -> None:
    result = await primes_range(10, 30)

    assert result["primes"] == [11, 13, 17, 19, 23, 29]


@pytest.mark.asyncio
async def test_primes_range_first_n_mode() -> None:
    result = await primes_range(5, 0)

    assert result["primes"] == [2, 3, 5, 7, 11]


@pytest.mark.asyncio
async def test_gcd_lcm_returns_bezout_coefficients() -> None:
    result = await gcd_lcm(["252", "105"], operation="gcd", extended=True)

    assert result["value"] == 21
    assert "bezout" in result


@pytest.mark.asyncio
async def test_lcm_multiple_numbers() -> None:
    result = await gcd_lcm([12, 18, 24], operation="lcm")

    assert result["value"] == 72


@pytest.mark.asyncio
async def test_euclidean_steps_are_recorded() -> None:
    result = await gcd_lcm([252, 105], operation="euclidean_steps")

    assert result["steps"][0] == "252 = 2×105 + 42"


@pytest.mark.asyncio
async def test_modular_inverse_returns_verification() -> None:
    result = await modular("inverse", "7", mod="26")

    assert result["value"] == 15
    assert result["verification"] == "7×15 mod 26 = 1"


@pytest.mark.asyncio
async def test_modular_power() -> None:
    result = await modular("power", "3", mod="7", power="100")

    assert result["value"] == pow(3, 100, 7)


@pytest.mark.asyncio
async def test_combinatorics_combinations_returns_expected_value() -> None:
    result = await combinatorics("combinations", 52, 5, as_formula=True)

    assert result["value"] == 2598960
    assert result["formula"] == "52!/(5!×47!)"


@pytest.mark.asyncio
async def test_combinatorics_catalan() -> None:
    result = await combinatorics("catalan", 5)

    assert result["value"] == 42


@pytest.mark.asyncio
async def test_partitions_count_returns_expected_value() -> None:
    result = await partitions(10)

    assert result["value"] == 42


@pytest.mark.asyncio
async def test_partitions_exact_number_of_parts() -> None:
    result = await partitions(5, operation="count_parts", k=2)

    assert result["value"] == 2


@pytest.mark.asyncio
async def test_diophantine_linear_equation_returns_small_solutions() -> None:
    result = await diophantine("3*x + 5*y = 7", bound=5)

    assert result["parametric_variable"] == "t"
    assert [4, -1] in result["small_solutions"]


@pytest.mark.asyncio
async def test_diophantine_no_solution_case() -> None:
    result = await diophantine("2*x + 4*y = 3")

    assert result["result"] == "no integer solutions"