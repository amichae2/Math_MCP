"""Error handling utilities for math tools."""

import functools
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)


def format_error(tool_name: str, error: Exception) -> dict[str, Any]:
    """Convert an exception into a user-facing error response."""
    logger.exception("Error in tool '%s': %s", tool_name, error)
    return {
        "result": None,
        "error": f"{type(error).__name__}: {error}",
        "tool": tool_name,
        "latex": None,
        "steps": None,
    }


def tool_error_handler(tool_name: str) -> Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]]:
    """Decorator that wraps tool functions with error handling and timing."""

    def decorator(
        func: Callable[..., Awaitable[dict[str, Any]]],
    ) -> Callable[..., Awaitable[dict[str, Any]]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            t0 = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                dt = time.perf_counter() - t0
                logger.info("tool=%s elapsed=%.4fs status=ok", tool_name, dt)
                return result
            except Exception as error:
                dt = time.perf_counter() - t0
                logger.info("tool=%s elapsed=%.4fs status=error", tool_name, dt)
                return format_error(tool_name, error)

        return wrapper

    return decorator