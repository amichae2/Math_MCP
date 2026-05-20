FROM python:3.12-slim

# Install uv for faster package installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy project metadata first for better cache behavior.
COPY pyproject.toml README.md ./
COPY src/ src/

RUN uv pip install --system -e ".[gpu]"

COPY tests/ tests/

CMD ["python", "-m", "math_mcp.server"]