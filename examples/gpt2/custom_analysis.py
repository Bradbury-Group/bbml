import numpy as np
from bbml.foundations.gpt2.datamodels import GPTConfig
from bbml.foundations.gpt2.gpt2_foundation import GPT2Foundation
from bbml.analysis import (
    get_adapter,
    get_metric,
    compute_similarity_matrix,
    compute_layer_statistics,
    find_redundant_pairs,
    compute_per_layer_mean_correlation,
)
import torch


def analyze_within_layer_similarity(foundation, adapter, device):
    """Analyze similarity of heads within each layer."""
    print("\n" + "=" * 70)
    print("Within-Layer Head Similarity Analysis")
    print("=" * 70 + "\n")
    
    index = adapter.extract_index(include_heads=True, include_full=False, include_ffn=False)
    
    metric = get_metric("cosine")
    
    # Define a function to get Q-heads for a specific layer
    def get_layer_q_heads(layer):
        return index.select(kind="attn.q.head", layer=layer)
    
    # Compute layer statistics
    results = compute_layer_statistics(
        units=index.select(kind="attn.q.head"),
        metric=metric,
        layer_fn=get_layer_q_heads,
        symmetric=True,
    )
    
    # Print results for each layer
    for result in results:
        layer = result["layer"]
        mean_sim = result["mean"]
        max_sim = result["max"]
        min_sim = result["min"]
        print(f"Layer {layer:2d}: mean={mean_sim:.3f}, max={max_sim:.3f}, min={min_sim:.3f}")
    
    # Find most/least diverse layers
    print("\n" + "-" * 70)
    least_diverse = results[-1]
    most_diverse = results[0]
    
    print("Most diverse layer (lowest mean similarity):")
    print(f"  Layer {most_diverse['layer']}: {most_diverse['mean']:.3f}")
    
    print("\nMost similar layer (highest mean similarity):")
    print(f"  Layer {least_diverse['layer']}: {least_diverse['mean']:.3f}")
    print()


def find_redundant_head_pairs(foundation, adapter, device):
    """Find the most similar head pairs across all layers."""
    print("\n" + "=" * 70)
    print("Finding Most Redundant Head Pairs")
    print("=" * 70 + "\n")
    
    index = adapter.extract_index(include_heads=True, include_full=False, include_ffn=False)
    
    # Get all Q-heads
    q_heads = index.select(kind="attn.q.head")
    print(f"Analyzing {len(q_heads)} Q-heads...")
    
    metric = get_metric("cosine")
    sim_matrix = compute_similarity_matrix(q_heads, metric, symmetric=True, show_progress=True)
    
    # Find top 10 most similar pairs
    pairs = find_redundant_pairs(q_heads, sim_matrix, k=10)
    
    print("\nTop 10 Most Similar Head Pairs:\n")
    print(f"{'Rank':<6} {'Head 1':<30} {'Head 2':<30} {'Similarity':<12}")
    print("-" * 80)
    
    for rank, (head1, head2, similarity) in enumerate(pairs, 1):
        print(f"{rank:<6} {head1.key:<30} {head2.key:<30} {similarity:.4f}")
    
    print("\nInterpretation:")
    print("  - Pairs with similarity > 0.9 are strong candidates for weight sharing")
    print("  - Consider grouping these heads before compression")
    print()


def compare_qk_correlation(foundation, adapter, device):
    """Analyze correlation between Q and K weights."""
    print("\n" + "=" * 70)
    print("Q-K Weight Correlation Analysis")
    print("=" * 70 + "\n")
    
    index = adapter.extract_index(include_heads=True, include_full=False, include_ffn=False)
    
    # Get all Q and K heads
    q_heads = index.select(kind="attn.q.head")
    k_heads = index.select(kind="attn.k.head")
    
    metric = get_metric("cosine")
    
    # Compute per-layer mean correlation
    layer_correlations = compute_per_layer_mean_correlation(q_heads, k_heads, metric)
    
    # Print results for each layer
    for result in layer_correlations:
        layer = result["layer"]
        mean_corr = result["mean_similarity"]
        print(f"Layer {layer:2d}: mean Q-K correlation = {mean_corr:.3f}")
    
    overall_mean = np.mean([r["mean_similarity"] for r in layer_correlations])
    print(f"\nOverall mean Q-K correlation: {overall_mean:.3f}")
    print("\nInterpretation:")
    print("  - High Q-K correlation suggests potential for joint factorization")
    print("  - Low correlation indicates Q and K learn different features")
    print()


def main():
    """Run all custom analyses."""
    print("\n" + "=" * 70)
    print("Custom Analysis Workflows with BBML")
    print("=" * 70)
    
    # Load model
    print("\nLoading GPT-2 model...")
    model = "gpt2"
    gpt_cfg = GPTConfig(from_hf=model)
    foundation = GPT2Foundation(gpt_cfg, train_config=None)
    device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    foundation.to(device=device)
    print(f"Model loaded on device: {device}\n")
    
    # Create adapter and load weights
    adapter = get_adapter("gpt2").load(foundation, device=device)
    
    # Run analyses
    analyze_within_layer_similarity(foundation, adapter, device)
    find_redundant_head_pairs(foundation, adapter, device)
    compare_qk_correlation(foundation, adapter, device)
    
    print("=" * 70)
    print("Analysis Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
