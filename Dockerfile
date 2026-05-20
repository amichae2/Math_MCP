FROM python:3.12-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/

RUN uv pip install --system -e .

CMD ["python", "-m", "math_mcp.server"]