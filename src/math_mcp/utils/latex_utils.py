"""LaTeX and rendering utilities."""

import base64
import io
from typing import Any

import sympy


def safe_latex(expr: Any) -> str:
    """Convert expression to LaTeX, falling back to str on failure."""
    try:
        return sympy.latex(expr)
    except Exception:
        return str(expr)


def latex_to_image_datauri(
    latex_str: str,
    fontsize: int = 14,
    dpi: int = 150,
) -> str | None:
    """
    Render LaTeX to a base64-encoded PNG data URI.

    Uses matplotlib's built-in mathtext renderer (no LaTeX installation needed).
    Returns None if rendering fails.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        text = ax.text(
            0.5,
            0.5,
            f"${latex_str}$",
            fontsize=fontsize,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        fig.canvas.draw()
        bbox = text.get_window_extent(renderer=fig.canvas.get_renderer())
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

        plt.close(fig)

        fig, ax = plt.subplots(figsize=(bbox.width + 0.2, bbox.height + 0.2))
        ax.text(
            0.5,
            0.5,
            f"${latex_str}$",
            fontsize=fontsize,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")

        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.1,
            transparent=True,
        )
        plt.close(fig)
        buffer.seek(0)

        encoded = base64.b64encode(buffer.read()).decode()
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return None