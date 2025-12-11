"""Helpers for writing reproducible measurement logs and figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from myproject.reporting.datamodels import Measurement

try:  # Matplotlib is optional but recommended for consistent figures
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - fallback when matplotlib unavailable
    plt = None


def save_measurements(
    measurements: Sequence[Measurement],
    report_dir: Path,
    name: str,
    *,
    title: str | None = None,
) -> dict[str, Path | None]:
    """Persist measurements to disk and optionally render a bar chart.

    Returns paths to the JSON artifact and generated figure (if matplotlib is installed).
    """

    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    json_path = report_path / f"{name}_measurements.json"
    json_payload = [measurement.model_dump() for measurement in measurements]
    json_path.write_text(json.dumps(json_payload, indent=2))

    figure_path: Path | None = None
    if measurements and plt is not None:
        figure_path = report_path / f"{name}_measurements.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            [m.name for m in measurements],
            [m.value for m in measurements],
            yerr=[m.std or 0.0 for m in measurements],
            color="#4C72B0",
            alpha=0.85,
            capsize=6,
        )
        ax.set_ylabel("value")
        ax.set_title(title or name)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=200)
        plt.close(fig)

    return {"json": json_path, "figure": figure_path}


__all__ = ["save_measurements"]
