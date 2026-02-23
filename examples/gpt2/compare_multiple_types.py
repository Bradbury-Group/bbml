from pathlib import Path
from bbml.foundations.gpt2.datamodels import GPTConfig
from bbml.foundations.gpt2.gpt2_foundation import GPT2Foundation
from bbml.analysis import get_adapter, get_metric
from bbml.analysis import compute_similarity_matrix, generate_report
import torch


DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent / "output"


def compare_cross_layer_heads(foundation, adapter, device, output_dir: Path = DEFAULT_OUTPUT_DIR):
    """Compare attention heads from different layers."""
    print("\n" + "=" * 70)
    print("Cross-Layer Head Comparison")
    print("=" * 70)
    
    # Extract weights
    index = adapter.extract_index(include_heads=True, include_full=False, include_ffn=False)
    
    # Get first 3 layers, all Q-heads
    q_heads = []
    for layer in range(3):
        q_heads.extend(index.select(kind="attn.q.head", layer=layer))
    
    print(f"Comparing {len(q_heads)} Q-heads from layers 0-2")
    
    # Compute similarity
    metric = get_metric("cosine")
    similarity_matrix = compute_similarity_matrix(q_heads, metric)
    
    # Generate report
    generate_report(
        similarity_matrix=similarity_matrix,
        units=q_heads,
        output_dir=str(output_dir),
        report_name="cross_layer_q_heads",
        metric_name="Cosine Similarity",
        title="Cross-Layer Q-Head Similarity (Layers 0-2)"
    )


def compare_qkv_matrices(foundation, adapter, device, output_dir: Path = DEFAULT_OUTPUT_DIR):
    """Compare full Q, K, V matrices across layers."""
    print("\n" + "=" * 70)
    print("Q/K/V Matrix Comparison")
    print("=" * 70)
    
    # Extract weights
    index = adapter.extract_index(include_heads=False, include_full=True, include_ffn=False)
    
    # Get all Q, K, V full matrices
    qkv_units = (
        index.select(kind="attn.q.full") +
        index.select(kind="attn.k.full") +
        index.select(kind="attn.v.full")
    )
    
    print(f"Comparing {len(qkv_units)} Q/K/V matrices")
    
    # Compute similarity
    metric = get_metric("cosine")
    similarity_matrix = compute_similarity_matrix(qkv_units, metric)
    
    # Generate report
    generate_report(
        similarity_matrix=similarity_matrix,
        units=qkv_units,
        output_dir=str(output_dir),
        report_name="qkv_matrices",
        metric_name="Cosine Similarity",
        title="Q/K/V Full Matrix Similarity",
        figsize=(16, 14)
    )


def compare_ffn_projections(foundation, adapter, device, output_dir: Path = DEFAULT_OUTPUT_DIR):
    """Compare FFN up and down projections."""
    print("\n" + "=" * 70)
    print("FFN Projection Comparison")
    print("=" * 70)
    
    # Extract weights
    index = adapter.extract_index(include_heads=False, include_full=False, include_ffn=True)
    
    # Get first 6 layers of FFN weights
    ffn_units = []
    for layer in range(6):
        ffn_units.extend(index.select(kind="ffn.up", layer=layer))
        ffn_units.extend(index.select(kind="ffn.down", layer=layer))
    
    print(f"Comparing {len(ffn_units)} FFN projections from layers 0-5")
    
    # Compute similarity
    metric = get_metric("cosine")
    similarity_matrix = compute_similarity_matrix(ffn_units, metric)
    
    # Generate report
    generate_report(
        similarity_matrix=similarity_matrix,
        units=ffn_units,
        output_dir=str(output_dir),
        report_name="ffn_projections",
        metric_name="Cosine Similarity",
        title="FFN Projection Similarity (Layers 0-5)"
    )


def main(output_dir: Path = DEFAULT_OUTPUT_DIR):
    """Run all comparison examples."""
    print("\n" + "=" * 70)
    print("GPT-2 Weight Comparison Examples")
    print("=" * 70)
    
    # Load model once for all analyses
    print("Loading GPT-2 model...")
    model = "gpt2"
    gpt_cfg = GPTConfig(from_hf=model)
    foundation = GPT2Foundation(gpt_cfg, train_config=None)
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    foundation.to(device=device)
    
    # Create adapter and load weights once
    adapter = get_adapter("gpt2")
    adapter.load(foundation, device=device)
    
    print(f"Model loaded on device: {device}\n")
    
    # Run each comparison
    compare_cross_layer_heads(foundation, adapter, device, output_dir=output_dir)
    compare_qkv_matrices(foundation, adapter, device, output_dir=output_dir)
    compare_ffn_projections(foundation, adapter, device, output_dir=output_dir)
    
    print("\n" + "=" * 70)
    print("All comparisons complete!")
    print(f"Check {output_dir}/ for results")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
