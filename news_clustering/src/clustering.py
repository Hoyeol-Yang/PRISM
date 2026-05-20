"""
Clustering module.
- Density-based clustering using HDBSCAN.
"""

import numpy as np
import hdbscan
from typing import Tuple, Optional


def cluster_articles(embeddings: np.ndarray,
                     min_cluster_size: int = 10,
                     min_samples: int = 5,
                     metric: str = 'euclidean') -> Tuple[np.ndarray, hdbscan.HDBSCAN]:
    """
    Cluster articles using HDBSCAN.

    Args:
        embeddings: Embedding array (N x embedding_dim)
        min_cluster_size: Minimum cluster size
        min_samples: Core point threshold
        metric: Distance metric

    Returns:
        (cluster label array, HDBSCAN model object)
        Label -1 indicates noise (not assigned to any cluster).
    """
    print(f"Starting HDBSCAN clustering...")
    print(f"  - min_cluster_size: {min_cluster_size}")
    print(f"  - min_samples: {min_samples}")
    print(f"  - metric: {metric}")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1  # use all CPU cores
    )

    cluster_labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f"\nClustering results:")
    print(f"  - Clusters: {n_clusters}")
    print(f"  - Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")

    return cluster_labels, clusterer


def get_cluster_stats(labels: np.ndarray) -> dict:
    """Return cluster statistics."""
    unique, counts = np.unique(labels, return_counts=True)
    stats = dict(zip(unique.astype(int), counts.astype(int)))
    return stats


if __name__ == "__main__":
    # Test with random data
    np.random.seed(42)
    fake_embeddings = np.random.rand(100, 384)

    labels, model = cluster_articles(fake_embeddings, min_cluster_size=5)
    print(f"\nCluster distribution: {get_cluster_stats(labels)}")
