#!/usr/bin/env python3
"""
Math MCP Server - Extremely powerful mathematics server.

Provides tools for symbolic math (SymPy), numerical computation (SciPy/NumPy),
linear algebra, statistics, discrete math, graph theory, and GPU acceleration.

Usage:
    uv run python -m math_mcp.server
    # Or via the entry point:
    math-mcp-server
"""

import asyncio
import logging
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import hw_config
from .tools import discrete, gpu, graphs, linalg, numerical, rendering, statistics, symbolic

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging to stderr (not stdout - that's the MCP transport)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(handler)


def register_all_tools(server: Server) -> None:
    """Register all tool families with the MCP server."""
    symbolic.register(server)
    numerical.register(server)
    linalg.register(server)
    statistics.register(server)
    discrete.register(server)
    graphs.register(server)
    gpu.register(server, hw_config)
    rendering.register(server)


async def serve() -> None:
    """Run the Math MCP server over stdio transport."""
    setup_logging()
    logger.info("%s", "=" * 60)
    logger.info("Math MCP Server starting...")
    logger.info("Hardware: %s", hw_config.to_dict())
    logger.info("%s", "=" * 60)

    server = Server("math-mcp-server")
    register_all_tools(server)

    async with stdio_server() as (reader, writer):
        logger.info("MCP transport established (stdio)")
        await server.run(reader, writer, server.create_initialization_options())


def main() -> None:
    """Synchronous entry point for console scripts and MCP hosts."""
    asyncio.run(serve())


def run() -> None:
    """Backward-compatible alias for the console entry point."""
    main()


if __name__ == "__main__":
    main()