from bbml.analysis.weights.units import WeightUnit, WeightIndex
from bbml.analysis.similarity import compute_similarity_matrix
from bbml.analysis.report import generate_report
from bbml.analysis.extractors.base import WeightExtractor, ModelAdapter
from bbml.analysis.utils import (
    compute_layer_statistics,
    find_redundant_pairs,
    compare_weight_correlation,
    compute_per_layer_mean_correlation,
)
from bbml.registries import WeightExtractorRegistry, MetricRegistry

import bbml.analysis.register_defaults


def get_adapter(name: str):
    """
    Get a weight extractor adapter by name.
    
    Args:
        name: The name of the adapter (e.g., "gpt2")
    
    Returns:
        An instantiated WeightExtractor subclass
    """
    extractor_class = WeightExtractorRegistry.get(name)
    return extractor_class()


def get_metric(name: str):
    """
    Get a metric by name.
    
    Args:
        name: The name of the metric (e.g., "cosine")
    
    Returns:
        An instantiated Metric subclass
    """
    metric_class = MetricRegistry.get(name)
    return metric_class()


__all__ = [
    "WeightUnit",
    "WeightIndex",
    "compute_similarity_matrix",
    "generate_report",
    "get_adapter",
    "get_metric",
    "WeightExtractor",
    "ModelAdapter",
    "compute_layer_statistics",
    "find_redundant_pairs",
    "compare_weight_correlation",
    "compute_per_layer_mean_correlation",
]
