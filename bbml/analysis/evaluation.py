from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

from bbml.analysis.weights.units import WeightUnit


@dataclass
class EvaluationResult:
    """Output of a pipeline evaluation.

    Attributes:
        score: A scalar summary score (e.g. perplexity, accuracy).
        metrics: Named metric values produced during evaluation.
        details: Arbitrary extra information (e.g. per-sample scores,
            benchmark breakdowns).
    """
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class PipelineEvaluator(ABC):
    """Evaluates the quality of a compressed model against a baseline.

    Supports both intrinsic metrics (perplexity, similarity reports) and
    extrinsic metrics (downstream benchmarks via lm-evaluation-harness).
    """

    @abstractmethod
    def evaluate(
        self,
        compressed: Union[Any, List[WeightUnit]],
        baseline: Union[Any, List[WeightUnit]],
    ) -> EvaluationResult:
        """Compare a compressed model or layer list against a reference.

        Args:
            compressed: The compressed model or list of compressed weight
                units to evaluate.
            baseline: The original (uncompressed) model or weight units
                used as the reference.

        Returns:
            An ``EvaluationResult`` with a scalar score and optional
            detailed metrics.
        """
        pass
