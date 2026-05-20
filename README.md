# Math MCP Server

Math MCP Server is a Python 3.12 Model Context Protocol server that exposes a broad mathematics toolkit over stdio. It combines symbolic algebra, calculus, numerical methods, linear algebra, statistics, discrete mathematics, graph theory, rendering, and optional GPU acceleration behind MCP tools that are usable from Claude Desktop, GitHub Copilot, and other MCP-compatible hosts.

The server is designed to run on CPU-only systems by default. If CUDA and CuPy are available, the GPU tool family accelerates supported workloads automatically and otherwise falls back to CPU implementations when possible.

## Installation

Editable install with `uv`:

```bash
uv pip install -e .
```

Editable install with `pip`:

```bash
pip install -e .
```

For development dependencies and tests:

```bash
.venv/bin/uv sync --extra dev
```

## Running The Server

Run directly with Python:

```bash
python -m math_mcp.server
```

Run through `uv`:

```bash
uv run python -m math_mcp.server
```

Run via the console script:

```bash
math-mcp-server
```

The server communicates over stdio, which is the expected transport for most MCP hosts.

## MCP Host Configuration

Most MCP clients accept a JSON config that declares a named server, the command to run, and its arguments. A generic config looks like this:

```json
{
	"mcpServers": {
		"math": {
			"command": "/absolute/path/to/.venv/bin/python",
			"args": ["-m", "math_mcp.server"],
			"cwd": "/absolute/path/to/math-mcp-server"
		}
	}
}
```

If the package is already installed in the active environment, you can use the console script instead:

```json
{
	"mcpServers": {
		"math": {
			"command": "math-mcp-server"
		}
	}
}
```

### Claude Desktop

Add the `math` server entry to your Claude Desktop MCP configuration file and point it at either the environment Python executable or the installed `math-mcp-server` script.

### GitHub Copilot

Use the same `mcpServers` structure in your MCP-enabled Copilot or VS Code host configuration. The server name can remain `math` so it matches the packaged MCP entry point.

### Other MCP Hosts

Any host that supports stdio MCP servers can run this project with the same command and args shown above.

## Tool Catalog

### Symbolic Tools

| Tool | Description |
| --- | --- |
| `simplify` | Simplify algebraic, trigonometric, and rational expressions. |
| `expand` | Expand products, powers, and composite symbolic expressions. |
| `factor` | Factor polynomials and integers into structured symbolic factors. |
| `solve` | Solve equations, systems, and inequalities over configurable domains. |
| `symbolic_integrate` | Compute indefinite and definite symbolic integrals. |
| `symbolic_diff` | Differentiate expressions, including higher-order derivatives. |
| `limit` | Evaluate one-sided and two-sided limits. |
| `series` | Produce Taylor or Laurent-like series expansions. |
| `symbolic_sum` | Evaluate finite and infinite symbolic summations. |
| `symbolic_product` | Evaluate finite and infinite symbolic products. |
| `dsolve` | Solve ordinary differential equations symbolically. |
| `laplace_transform` | Compute Laplace transforms in the frequency domain. |
| `inverse_laplace` | Transform Laplace-domain expressions back to time-domain forms. |

### Numerical Tools

| Tool | Description |
| --- | --- |
| `numerical_integrate` | Numerically integrate real-valued functions on finite or improper intervals. |
| `find_root` | Find roots with bracketing or open methods such as Brent or Newton. |
| `ode_solve` | Solve initial value problems for scalar or vector ODE systems. |
| `interpolate_data` | Interpolate tabular data with linear or spline-based methods. |
| `curve_fit` | Fit parametric models to observed data and report fitted parameters. |

### Linear Algebra Tools

| Tool | Description |
| --- | --- |
| `matrix_multiply` | Multiply dense matrices and vectors. |
| `solve_linear` | Solve square linear systems with direct or iterative methods. |
| `lstsq` | Solve overdetermined systems with least-squares regression. |
| `matrix_decompose` | Compute QR, LU, Cholesky, or SVD-style decompositions. |
| `eigen_decomp` | Compute eigenvalues and eigenvectors. |
| `matrix_info` | Report determinant, rank, conditioning, definiteness, and shape metadata. |

### Statistics Tools

| Tool | Description |
| --- | --- |
| `describe` | Return descriptive statistics for a sample. |
| `distribution` | Evaluate distribution PDFs, CDFs, quantiles, and related functions. |
| `hypothesis_test` | Run supported hypothesis tests and report significance details. |
| `regression` | Fit linear and polynomial regression models. |
| `bootstrap` | Estimate confidence intervals with bootstrap resampling. |
| `random_sample` | Draw samples from supported probability distributions. |

### Discrete Math And Number Theory Tools

| Tool | Description |
| --- | --- |
| `factor_int` | Factor integers into prime powers. |
| `is_prime` | Test whether an integer is prime. |
| `primes_range` | Enumerate primes over an interval or generate the first `n` primes. |
| `gcd_lcm` | Compute GCD, LCM, extended GCD, or Euclidean algorithm steps. |
| `modular` | Solve modular arithmetic tasks such as inverses, powers, and congruences. |
| `combinatorics` | Evaluate permutations, combinations, Catalan numbers, and related counts. |
| `partitions` | Count integer partitions and constrained partition variants. |
| `diophantine` | Solve supported Diophantine equations over the integers. |

### Graph Theory Tools

| Tool | Description |
| --- | --- |
| `graph_create` | Build graphs from named templates, edge lists, or adjacency matrices. |
| `shortest_path` | Compute shortest paths on weighted or unweighted graphs. |
| `spanning_tree` | Compute a spanning tree or minimum spanning tree. |
| `graph_metrics` | Report graph connectivity and structural metrics. |
| `max_flow` | Compute maximum flow on directed graphs. |

### GPU Tools

| Tool | Description |
| --- | --- |
| `gpu_matrix_multiply` | Accelerate matrix multiplication with GPU support and CPU fallback. |
| `gpu_fft` | Compute FFTs on GPU when available. |
| `gpu_eigen_batch` | Compute batched eigenvalues for multiple matrices. |
| `gpu_solve` | Solve batched linear systems using GPU acceleration where available. |

### Rendering Tools

| Tool | Description |
| --- | --- |
| `to_latex` | Convert symbolic expressions to LaTeX and renderable output. |
| `render_math` | Render LaTeX math to an image data URI. |
| `plot_function` | Plot explicit functions over configurable domains. |
| `plot_implicit` | Plot implicit equations and inequalities. |

## Hardware Requirements

- Python 3.12 or newer.
- CPU-only systems are fully supported for the non-GPU tool families.
- Optional CUDA-capable NVIDIA GPU for the GPU tool family.
- If CUDA or CuPy is unavailable, GPU tools either fall back to CPU implementations or report that GPU acceleration is not available.
- Memory needs depend on workload size. Large symbolic expressions, large matrices, and graph algorithms can require significant RAM.

## Example Queries

- Simplify `(x^2 - 1) / (x - 1)`.
- Solve `x^2 - 5x + 6 = 0` over the reals.
- Compute the definite integral of `sin(x)` from `0` to `pi`.
- Find the eigenvalues of `[[4, 1], [1, 3]]`.
- Fit an exponential decay model to observed time-series data.
- Compute the bootstrap confidence interval for a sample mean.
- Factor `123456789` into prime powers.
- Find the shortest weighted path from node `A` to node `D`.
- Render the equation `x^2 + y^2 = 1` as an implicit plot.
- Multiply a batch of matrices with GPU acceleration if hardware is available.

## Verification

```bash
uv run python -c "from math_mcp.server import main; print('Import OK')"
uv run python -c "from math_mcp.tools.symbolic import simplify; print('Tools OK')"
uv run pytest tests/ -v
```
