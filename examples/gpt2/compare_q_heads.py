from pathlib import Path
from bbml.foundations.gpt2.datamodels import GPTConfig
from bbml.foundations.gpt2.gpt2_foundation import GPT2Foundation
from bbml.analysis import get_adapter, get_metric
from bbml.analysis import compute_similarity_matrix, generate_report
import torch

print("=" * 70)
print("GPT-2 Q-Head Similarity Analysis")
print("=" * 70)
print()

print("Step 1: Loading GPT-2 model...")
model = "gpt2"
gpt_cfg = GPTConfig(from_hf=model)
foundation = GPT2Foundation(gpt_cfg, train_config=None)
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
foundation.to(device=device)
print()

print("Step 2: Extracting weight units...")
adapter = get_adapter("gpt2")
adapter.load(foundation, device=device)

index = adapter.extract_index(
    include_heads=True,
    include_full=False,
    include_ffn=False
)

q_heads = index.select(kind="attn.q.head")
print(f"Found {len(q_heads)} Q-head units")
print(f"First few: {[u.key for u in q_heads[:3]]}")
print()

print("Step 3: Computing pairwise cosine similarity...")
metric = get_metric("cosine")

similarity_matrix = compute_similarity_matrix(
    units=q_heads,
    metric=metric,
    symmetric=True,
    show_progress=True
)
print(f"Computed {len(q_heads)}x{len(q_heads)} similarity matrix")
print()

print("Step 4: Generating visualization and report...")
output_dir = Path(__file__).parent.parent.parent / "output"

report_info = generate_report(
    similarity_matrix=similarity_matrix,
    units=q_heads,
    output_dir=str(output_dir),
    report_name="gpt2_q_heads_similarity",
    metric_name="Cosine Similarity",
    title="GPT-2 Q-Head Weight Similarity",
    figsize=(14, 12),
    vmin=-1.0,
    vmax=1.0
)

print()
print("=" * 70)
print("Analysis Complete!")
print("=" * 70)
print(f"Report: {report_info['report_path']}")
print(f"Heatmap: {report_info['image_path']}")
print()
print("Summary statistics:")
for key, value in report_info['statistics'].items():
    print(f"  {key:>10s}: {value:.4f}")
print()
print("Interpretation:")
print("  - High similarity (>0.8) suggests potential for weight sharing")
print("  - Low similarity (<0.3) indicates diverse representations")
print("  - Block patterns may reveal head groupings")
print()
