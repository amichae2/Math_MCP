"""Tests for discrete math and number theory tools."""

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


# --- factor_int ---


@pytest.mark.asyncio
async def test_factor_int_12() -> None:
    result = await factor_int(n="12")
    assert result["factors"] == [[2, 2], [3, 1]]
    assert result["is_complete"] is True


@pytest.mark.asyncio
async def test_factor_int_prime() -> None:
    result = await factor_int(n="97")
    assert result["factors"] == [[97, 1]]


@pytest.mark.asyncio
async def test_factor_int_one() -> None:
    result = await factor_int(n="1")
    assert result["factors"] == []
    assert result["is_complete"] is True


@pytest.mark.asyncio
async def test_factor_int_large() -> None:
    result = await factor_int(n="123456789")
    factor_primes = [pair[0] for pair in result["factors"]]
    assert 3 in factor_primes
    assert 3607 in factor_primes
    assert 3803 in factor_primes


# --- is_prime ---


@pytest.mark.asyncio
async def test_is_prime_true() -> None:
    result = await is_prime(n="97")
    assert result["is_prime"] is True


@pytest.mark.asyncio
async def test_is_prime_false() -> None:
    result = await is_prime(n="100")
    assert result["is_prime"] is False


@pytest.mark.asyncio
async def test_is_prime_probabilistic() -> None:
    result = await is_prime(n="97", rigorous=False)
    assert result["is_prime"] is True
    assert result["method"] == "Miller-Rabin"


# --- primes_range ---


@pytest.mark.asyncio
async def test_primes_range_small() -> None:
    result = await primes_range(start=10, end=30)
    assert result["primes"] == [11, 13, 17, 19, 23, 29]


@pytest.mark.asyncio
async def test_primes_range_first_n() -> None:
    result = await primes_range(start=5, end=0)
    assert result["primes"] == [2, 3, 5, 7, 11]


# --- gcd_lcm ---


@pytest.mark.asyncio
async def test_gcd() -> None:
    result = await gcd_lcm(numbers=[252, 105])
    assert result["value"] == 21


@pytest.mark.asyncio
async def test_gcd_extended() -> None:
    result = await gcd_lcm(numbers=[252, 105], extended=True)
    assert result["value"] == 21
    assert "bezout" in result
    assert 252 * result["bezout"]["x"] + 105 * result["bezout"]["y"] == 21


@pytest.mark.asyncio
async def test_lcm() -> None:
    result = await gcd_lcm(numbers=[12, 18, 24], operation="lcm")
    assert result["value"] == 72


# --- modular ---


@pytest.mark.asyncio
async def test_modular_inverse() -> None:
    result = await modular(operation="inverse", a="7", mod="26")
    assert result["value"] == 15
    assert (7 * 15) % 26 == 1


@pytest.mark.asyncio
async def test_modular_power() -> None:
    result = await modular(operation="power", a="3", mod="7", power="100")
    assert result["value"] == pow(3, 100, 7)


@pytest.mark.asyncio
async def test_modular_crt() -> None:
    """x congruent 2 mod 3, 3 mod 5, 2 mod 7 gives 23 mod 105."""
    result = await modular(operation="crt", a="2,3,2", mod="3,5,7")
    assert result["value"] == 23
    assert result["modulus"] == 105


# --- combinatorics ---


@pytest.mark.asyncio
async def test_combinations() -> None:
    result = await combinatorics(operation="combinations", n=10, k=5)
    assert result["value"] == 252


@pytest.mark.asyncio
async def test_permutations() -> None:
    result = await combinatorics(operation="permutations", n=5, k=3)
    assert result["value"] == 60


@pytest.mark.asyncio
async def test_factorial() -> None:
    result = await combinatorics(operation="factorial", n=5)
    assert result["value"] == 120


@pytest.mark.asyncio
async def test_fibonacci() -> None:
    result = await combinatorics(operation="fibonacci", n=10)
    assert result["value"] == 55


# --- partitions ---


@pytest.mark.asyncio
async def test_partitions_count() -> None:
    result = await partitions(n=10, operation="count")
    assert result["value"] == 42


@pytest.mark.asyncio
async def test_partitions_list_small() -> None:
    result = await partitions(n=4, operation="list")
    assert result["count"] == 5


@pytest.mark.asyncio
async def test_partitions_list_too_large() -> None:
    result = await partitions(n=100, operation="list")
    assert "warning" in result or "disabled" in str(result.get("result", ""))


# --- diophantine ---


@pytest.mark.asyncio
async def test_diophantine_linear() -> None:
    """3x + 5y = 7"""
    result = await diophantine(equation="3*x + 5*y = 7")
    assert "no integer solutions" not in str(result.get("result", ""))
    assert result.get("general_solution") is not None


@pytest.mark.asyncio
async def test_diophantine_pythagorean() -> None:
    """x squared plus y squared equals z squared."""
    result = await diophantine(equation="x**2 + y**2 - z**2")
    assert "no integer solutions" not in str(result.get("result", ""))
