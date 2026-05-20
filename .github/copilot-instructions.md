# Copilot Instructions for math-mcp-server

## Project Conventions

### All MCP Tools
- Must have complete type annotations (Python 3.12+ syntax)
- Use `async def` for all tool handler functions
- Return structured dicts: `{"result": ..., "latex": str | None, "steps": list[str] | None}`
- Log every tool call with timing to stderr: `logger.info(f"tool={name} elapsed={dt:.3f}s")`
- Validate all inputs before computation
- Catch ALL exceptions and return `{"error": str, "tool": str}` — never let exceptions propagate

### Code Style
- Single file per tool family in `src/math_mcp/tools/`
- Shared utilities in `src/math_mcp/utils/`
- Use `sympy.sympify(locals={...})` for all expression parsing — never raw `eval`
- All matrix inputs are `list[list[float]]` (row-major)
- Use `pydantic` for input validation where complex

### Error Messages
- User-facing: "Could not integrate: the expression contains an unsupported special function: ..."
- Never expose raw tracebacks to the user
- Log raw tracebacks to stderr with `logger.exception(...)`

### Testing
- Every tool must have a corresponding test file in `tests/`
- Use pytest with fixtures
- Test: happy path, edge cases, error cases, timeout cases