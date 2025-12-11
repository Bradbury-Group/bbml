"""Experiment lifecycle: load → optimize → run → report → cleanup."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Experiment metadata."""

    name: str
    report_dir: Path = Path("reports")
    seed: int = 42
    iterations: int = 1  # For benchmarking/sweep experiments
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class Experiment:
    """Base experiment. Override what you need.

    Lifecycle: load -> optimize -> run → report -> cleanup

    For iteration-based experiments (benchmarks, sweeps), override run_once()
    and let run() handle the loop. For single-run experiments, override run().
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.report_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> None:
        """Load data, models, etc."""
        pass

    def optimize(self) -> None:
        """Apply optimizations (torch.compile, quantization, etc.)."""
        pass

    def run_once(self, iteration: int = 0) -> Any:
        """Single iteration. Override for benchmarks/sweeps."""
        pass

    def run(self) -> None:
        """Main computation. Default: loop over run_once() for config.iterations."""
        for i in range(self.config.iterations):
            self.run_once(iteration=i)

    def report(self) -> None:
        """Save results."""
        pass

    def cleanup(self) -> None:
        """Release resources."""
        pass

    def run_all(self) -> None:
        """Execute: load -> optimize -> run -> report -> cleanup."""
        self.load()
        self.optimize()
        self.run()
        self.report()
        self.cleanup()
