from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from bbml.analysis.weights.units import WeightUnit


def generate_report(
    similarity_matrix: np.ndarray,
    units: List[WeightUnit],
    output_dir: str,
    report_name: str,
    metric_name: str = "Similarity",
    title: str = "Weight Similarity Matrix",
    figsize: Tuple[int, int] = (14, 12),
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_path = output_path / f"{report_name}.png"
    report_path = output_path / f"{report_name}.md"
    
    n = len(units)
    labels = [u.key for u in units]
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(similarity_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    
    ax.set_xlabel("Unit")
    ax.set_ylabel("Unit")
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_name)
    
    plt.tight_layout()
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    mask = np.ones_like(similarity_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    off_diag_values = similarity_matrix[mask]
    
    statistics = {
        "mean": float(np.mean(off_diag_values)),
        "median": float(np.median(off_diag_values)),
        "std": float(np.std(off_diag_values)),
        "min": float(np.min(off_diag_values)),
        "max": float(np.max(off_diag_values)),
    }
    
    max_idx = np.unravel_index(similarity_matrix[mask].argmax(), similarity_matrix.shape)
    min_idx = np.unravel_index(similarity_matrix[mask].argmin(), similarity_matrix.shape)
    
    most_similar = (units[max_idx[0]].key, units[max_idx[1]].key, similarity_matrix[max_idx])
    least_similar = (units[min_idx[0]].key, units[min_idx[1]].key, similarity_matrix[min_idx])
    
    with open(report_path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"![Similarity Heatmap]({image_path.name})\n\n")
        f.write(f"## Summary Statistics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        for key, value in statistics.items():
            f.write(f"| {key.capitalize()} | {value:.4f} |\n")
        f.write(f"\n## Analysis Details\n\n")
        f.write(f"- **Total Units**: {n}\n")
        f.write(f"- **Matrix Size**: {n}x{n}\n")
        f.write(f"- **Metric**: {metric_name}\n")
        f.write(f"\n## Most Similar Pair\n\n")
        f.write(f"- **Units**: `{most_similar[0]}` ↔ `{most_similar[1]}`\n")
        f.write(f"- **Similarity**: {most_similar[2]:.4f}\n")
        f.write(f"\n## Least Similar Pair\n\n")
        f.write(f"- **Units**: `{least_similar[0]}` ↔ `{least_similar[1]}`\n")
        f.write(f"- **Similarity**: {least_similar[2]:.4f}\n")
    
    return {
        "image_path": str(image_path),
        "report_path": str(report_path),
        "statistics": statistics,
        "most_similar": most_similar,
        "least_similar": least_similar,
    }
