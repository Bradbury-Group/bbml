# BBML Compression Quick Start

## Install

```bash
pip install -e .
```

## Run the Example

```bash
python examples/gpt2/compare_multiple_types.py
```

Results will be saved to `examples/gpt2/results/`.

---

## Extending the Examples

### Extracting Weight Matrices

Use `get_adapter` to load a model's weights into a `WeightIndex`, then use `.select()` to filter by `kind`, `layer`, or `head`.

```python
from bbml.analysis import get_adapter

adapter = get_adapter("gpt2")
adapter.load(foundation, device=device)

index = adapter.extract_index(
    include_heads=True,   # per-head tensors (attn.q.head, attn.k.head, attn.v.head)
    include_full=True,    # full projection matrices (attn.q.full, attn.k.full, attn.v.full)
    include_ffn=True,     # FFN weights (ffn.up, ffn.down)
)

# Filter by kind and layer
q_heads = index.select(kind="attn.q.head", layer=0)

# Inspect available kinds and layers
print(index.kinds())   # ['attn.k.full', 'attn.q.full', 'ffn.down', ...]
print(index.layers())  # [0, 1, 2, ...]
```

Each item in the index is a `WeightUnit` with `.key`, `.tensor`, `.kind`, `.layer`, and `.head` attributes.

To support a new model, subclass `WeightExtractor` and register it:

```python
from bbml.analysis.extractors.base import WeightExtractor
from bbml.analysis.weights.units import WeightUnit, WeightIndex
from bbml.registries import WeightExtractorRegistry

@WeightExtractorRegistry.register("my-model")
class MyModelExtractor(WeightExtractor):
    def load(self, model, device="cpu"):
        self.model = model
        return self

    def extract_index(self, include_heads=False, include_full=True, include_ffn=True):
        units = []
        # Populate with WeightUnit objects from your model's state dict
        units.append(WeightUnit(
            key="layer0.attn.q.full",
            tensor=self.model.layers[0].q_proj.weight.clone(),
            kind="attn.q.full",
            layer=0,
        ))
        return WeightIndex(units)

    def get_config(self):
        return {}
```

Then use it with `get_adapter("my-model")`.

---

### Creating Your Own Metric

Subclass `Metric`, implement `compare()`, and register it:

```python
import torch
from bbml.analysis.metrics.base import Metric, MetricResult
from bbml.registries import MetricRegistry

@MetricRegistry.register("dot-product")
class DotProductMetric(Metric):
    def __init__(self):
        super().__init__(name="dot-product")

    def compare(self, weight1: torch.Tensor, weight2: torch.Tensor) -> MetricResult:
        score = float(torch.dot(weight1.flatten(), weight2.flatten()).item())
        return MetricResult(score=score)
```

Then use it with `get_metric("dot-product")`.

---

### Computing a Similarity Matrix

Pass a list of `WeightUnit` objects and a metric to `compute_similarity_matrix`. It returns a symmetric `np.ndarray` of shape `(n, n)`.

```python
from bbml.analysis import compute_similarity_matrix, get_metric

metric = get_metric("cosine")
units = index.select(kind="attn.q.head", layer=0)

matrix = compute_similarity_matrix(units, metric)
# matrix[i, j] is the similarity score between units[i] and units[j]
```

---

### Generating Reports

`generate_report` saves a heatmap image and a Markdown summary to `output_dir`.

```python
from bbml.analysis import generate_report

generate_report(
    similarity_matrix=matrix,
    units=units,
    output_dir="output/my_analysis",
    report_name="q_heads_layer0",       # output filename (no extension)
    metric_name="Cosine Similarity",    # colorbar label
    title="Q-Head Similarity (Layer 0)",
    figsize=(12, 10),                   # optional, default (14, 12)
    vmin=-1.0,                          # optional colorscale bounds
    vmax=1.0,
    cmap="RdBu_r",                      # optional matplotlib colormap
)
```

Outputs written to `output_dir/`:
- `q_heads_layer0.png` — heatmap
- `q_heads_layer0.md` — summary statistics and most/least similar pairs


For full documentation, see the main README.