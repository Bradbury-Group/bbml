import torch

from bbml.analysis.metrics.base import Metric, MetricResult
from bbml.analysis.similarity import compute_similarity_matrix
from bbml.analysis.utils import compute_layer_statistics
from bbml.analysis.weights.units import WeightIndex, WeightUnit


class _AsymmetricMetric(Metric):
    def __init__(self):
        super().__init__(name="asymmetric")

    def compare(self, weight1: torch.Tensor, weight2: torch.Tensor) -> MetricResult:
        score = float(weight1.sum().item() - weight2.sum().item())
        return MetricResult(score=score)


class _EqualityMetric(Metric):
    def __init__(self):
        super().__init__(name="equality")

    def compare(self, weight1: torch.Tensor, weight2: torch.Tensor) -> MetricResult:
        return MetricResult(score=1.0 if torch.allclose(weight1, weight2) else 0.0)


def test_weight_index_uses_canonical_collection_api_only():
    index = WeightIndex([])

    assert hasattr(index, "kinds")
    assert hasattr(index, "layers")
    assert hasattr(index, "heads")

    assert not hasattr(index, "get_kinds")
    assert not hasattr(index, "get_layers")
    assert not hasattr(index, "get_heads")


def test_compute_similarity_matrix_supports_symmetric_and_non_symmetric():
    units = [
        WeightUnit(key="a", tensor=torch.tensor([1.0]), kind="k"),
        WeightUnit(key="b", tensor=torch.tensor([3.0]), kind="k"),
    ]

    metric = _AsymmetricMetric()

    symmetric_matrix = compute_similarity_matrix(units, metric, symmetric=True, show_progress=False)
    non_symmetric_matrix = compute_similarity_matrix(units, metric, symmetric=False, show_progress=False)

    assert symmetric_matrix.shape == (2, 2)
    assert symmetric_matrix[0, 0] == 1.0
    assert symmetric_matrix[1, 1] == 1.0
    assert symmetric_matrix[0, 1] == symmetric_matrix[1, 0]

    assert non_symmetric_matrix.shape == (2, 2)
    assert non_symmetric_matrix[0, 0] == 1.0
    assert non_symmetric_matrix[1, 1] == 1.0
    assert non_symmetric_matrix[0, 1] != non_symmetric_matrix[1, 0]


def test_compute_layer_statistics_uses_similarity_matrix_logic():
    units = [
        WeightUnit(key="layer0.a", tensor=torch.tensor([1.0]), kind="k", layer=0),
        WeightUnit(key="layer0.b", tensor=torch.tensor([2.0]), kind="k", layer=0),
        WeightUnit(key="layer1.a", tensor=torch.tensor([3.0]), kind="k", layer=1),
        WeightUnit(key="layer1.b", tensor=torch.tensor([3.0]), kind="k", layer=1),
    ]

    metric = _EqualityMetric()

    def layer_fn(layer: int):
        return [unit for unit in units if unit.layer == layer]

    stats = compute_layer_statistics(units, metric, layer_fn=layer_fn, symmetric=True)

    assert [row["layer"] for row in stats] == [0, 1]

    layer0 = stats[0]
    layer1 = stats[1]

    assert layer0["mean"] == 0.0
    assert layer0["max"] == 0.0
    assert layer0["min"] == 0.0

    assert layer1["mean"] == 1.0
    assert layer1["max"] == 1.0
    assert layer1["min"] == 1.0
