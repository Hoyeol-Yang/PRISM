"""
Evaluation module.
- Print cluster sample articles.
- Save clustering results in JSON format.
- Visualization (cluster size distribution, bias distribution).
"""

import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Optional
from collections import Counter


plt.rcParams['axes.unicode_minus'] = False

LABEL_NAMES_3 = {0: 'Left', 1: 'Center', 2: 'Right'}
LABEL_NAMES_5 = {0: 'Left', 1: 'Lean Left', 2: 'Center', 3: 'Lean Right', 4: 'Right'}

COLORS_3 = {'Left': '#3B82F6', 'Center': '#6B7280', 'Right': '#EF4444'}
COLORS_5 = {
    'Left': '#1D4ED8', 'Lean Left': '#60A5FA',
    'Center': '#6B7280',
    'Lean Right': '#F87171', 'Right': '#DC2626',
}


def _detect_label_names(df, label_col='label'):
    """Auto-select 3-class or 5-class mapping based on the label value range in the data."""
    max_label = int(df[label_col].max())
    if max_label >= 3:
        return LABEL_NAMES_5, COLORS_5
    return LABEL_NAMES_3, COLORS_3


def print_cluster_samples(df: pd.DataFrame,
                          text_col: str = 'title',
                          cluster_col: str = 'cluster',
                          label_col: str = 'label',
                          n_samples: int = 3,
                          n_clusters: int = 10) -> None:
    """Print sample articles per cluster."""
    label_names, _ = _detect_label_names(df, label_col)

    cluster_sizes = df[df[cluster_col] != -1][cluster_col].value_counts()
    top_clusters = cluster_sizes.head(n_clusters).index.tolist()

    print("=" * 80)
    print(f"Top {len(top_clusters)} clusters by size")
    print("=" * 80)

    for cluster_id in top_clusters:
        cluster_df = df[df[cluster_col] == cluster_id]
        bias_dist = cluster_df[label_col].value_counts()
        bias_str = ", ".join([f"{label_names.get(k, k)}: {v}" for k, v in bias_dist.items()])

        print(f"\n[Cluster {cluster_id}] ({len(cluster_df)} articles)")
        print(f"  Bias distribution: {bias_str}")
        print("-" * 60)

        samples = cluster_df.head(n_samples)
        for idx, row in samples.iterrows():
            bias = label_names.get(row[label_col], row[label_col])
            title = row[text_col][:80] + "..." if len(str(row[text_col])) > 80 else row[text_col]
            print(f"  [{bias}] {title}")

    noise_count = len(df[df[cluster_col] == -1])
    if noise_count > 0:
        print(f"\n[Noise] {noise_count} articles (not assigned to any cluster)")


def plot_cluster_sizes(df: pd.DataFrame,
                       cluster_col: str = 'cluster',
                       top_n: int = 30,
                       save_path: Optional[str] = None) -> None:
    """Visualize cluster size distribution (top N clusters)."""

    cluster_counts = df[df[cluster_col] != -1][cluster_col].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = plt.cm.viridis(np.linspace(0.8, 0.2, len(cluster_counts)))

    bars = ax.bar(range(len(cluster_counts)), cluster_counts.values, color=colors, edgecolor='white', linewidth=0.5)

    for i, (bar, val) in enumerate(zip(bars[:10], cluster_counts.values[:10])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Cluster Rank (by size)', fontsize=12)
    ax.set_ylabel('Number of Articles', fontsize=12)
    ax.set_title(f'Top {top_n} Clusters by Size', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, len(cluster_counts), 5))
    ax.set_xticklabels(range(1, len(cluster_counts)+1, 5))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    total_clustered = df[df[cluster_col] != -1].shape[0]
    total_noise = df[df[cluster_col] == -1].shape[0]
    n_clusters = df[df[cluster_col] != -1][cluster_col].nunique()

    stats_text = f"Total Clusters: {n_clusters} | Clustered: {total_clustered:,} | Noise: {total_noise:,}"
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes, ha='right', va='top',
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Graph saved: {save_path}")

    plt.close(fig)


def plot_bias_distribution(df: pd.DataFrame,
                           cluster_col: str = 'cluster',
                           label_col: str = 'label',
                           top_n: int = 20,
                           save_path: Optional[str] = None) -> None:
    """Visualize per-cluster bias distribution (stacked bar chart)."""

    label_names, colors = _detect_label_names(df, label_col)
    bias_order = [label_names[k] for k in sorted(label_names.keys())]

    cluster_sizes = df[df[cluster_col] != -1][cluster_col].value_counts()
    top_clusters = cluster_sizes.head(top_n).index.tolist()

    data = {'cluster': []}
    for name in bias_order:
        data[name] = []

    for cluster_id in top_clusters:
        cluster_df = df[df[cluster_col] == cluster_id]
        bias_counts = cluster_df[label_col].value_counts()
        data['cluster'].append(f"C{cluster_id}")
        for label_id, name in label_names.items():
            data[name].append(bias_counts.get(label_id, 0))

    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(plot_df))
    width = 0.7

    bottom = np.zeros(len(plot_df))
    for bias_name in bias_order:
        bars = ax.bar(x, plot_df[bias_name], width, label=bias_name, bottom=bottom,
                      color=colors[bias_name], edgecolor='white', linewidth=0.5)
        bottom += plot_df[bias_name].values

    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Number of Articles', fontsize=12)
    ax.set_title(f'Bias Distribution per Cluster (Top {top_n})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df['cluster'], rotation=45, ha='right')
    ax.legend(title='Bias', loc='upper right')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Graph saved: {save_path}")

    plt.close(fig)


def save_clusters_to_json(df: pd.DataFrame,
                          output_path: str,
                          cluster_col: str = 'cluster',
                          label_col: str = 'label') -> None:
    """Save clustering results in JSON format."""
    label_names, _ = _detect_label_names(df, label_col)

    result = {
        "metadata": {
            "total_articles": len(df),
            "n_clusters": len(df[df[cluster_col] != -1][cluster_col].unique()),
            "n_noise": len(df[df[cluster_col] == -1])
        },
        "clusters": []
    }

    cluster_sizes = df[df[cluster_col] != -1][cluster_col].value_counts()
    sorted_clusters = cluster_sizes.index.tolist()

    for cluster_id in sorted_clusters:
        cluster_df = df[df[cluster_col] == cluster_id]
        bias_dist = cluster_df[label_col].value_counts().to_dict()
        bias_dist_named = {label_names.get(k, str(k)): int(v) for k, v in bias_dist.items()}

        articles = []
        for idx, row in cluster_df.iterrows():
            articles.append({
                "title": row['title'],
                "text": row['text'],
                "bias": label_names.get(row[label_col], str(row[label_col]))
            })

        result["clusters"].append({
            "cluster_id": int(cluster_id),
            "size": len(cluster_df),
            "bias_distribution": bias_dist_named,
            "articles": articles
        })

    noise_df = df[df[cluster_col] == -1]
    if len(noise_df) > 0:
        noise_articles = []
        for idx, row in noise_df.iterrows():
            noise_articles.append({
                "title": row['title'],
                "text": row['text'],
                "bias": label_names.get(row[label_col], str(row[label_col]))
            })

        result["clusters"].append({
            "cluster_id": "noise",
            "size": len(noise_df),
            "bias_distribution": {
                label_names.get(k, str(k)): int(v)
                for k, v in noise_df[label_col].value_counts().to_dict().items()
            },
            "articles": noise_articles
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"JSON saved: {output_path}")


def get_clustering_summary(df: pd.DataFrame, cluster_col: str = 'cluster') -> dict:
    """Return summary statistics for clustering results."""
    labels = df[cluster_col].values
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    n_total = len(labels)

    cluster_sizes = df[df[cluster_col] != -1][cluster_col].value_counts()

    return {
        'total_articles': n_total,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / n_total,
        'avg_cluster_size': cluster_sizes.mean() if len(cluster_sizes) > 0 else 0,
        'max_cluster_size': cluster_sizes.max() if len(cluster_sizes) > 0 else 0,
        'min_cluster_size': cluster_sizes.min() if len(cluster_sizes) > 0 else 0,
    }


if __name__ == "__main__":
    import os

    if os.path.exists("outputs/clusters.csv"):
        df = pd.read_csv("outputs/clusters.csv")
        print_cluster_samples(df, n_samples=2, n_clusters=5)
        plot_cluster_sizes(df, save_path="outputs/cluster_sizes.png")
        plot_bias_distribution(df, save_path="outputs/bias_distribution.png")
