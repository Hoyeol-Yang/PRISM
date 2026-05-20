"""
Bias-aware similar article recommendation within clusters (AllSides-L only).

Recommends the Top-K most similar articles per bias class within the same cluster
using cosine similarity. Interoperable with IG_khan XAI results via shared article_id.

Prerequisites:
    1. Run scripts/add_predicted_labels.py (generates predicted_label)
    2. Run main.py (generates clustering results)

Usage:
    # Default: AllSides-L test set, Top-2 recommendations
    python recommend.py

    # Top-3 recommendations
    python recommend.py --top_k 3

    # Use a different SBERT model's results
    python recommend.py --model_name all-mpnet-base-v2
"""

import numpy as np
import pandas as pd
import json
import os
import sys
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

CLUSTERING_ROOT = os.path.dirname(os.path.abspath(__file__))
UNIFIED_ROOT = os.path.dirname(CLUSTERING_ROOT)
sys.path.insert(0, UNIFIED_ROOT)

from common.data_loader import BIAS_MAP_ALLSIDES_L

BIAS_MAP = BIAS_MAP_ALLSIDES_L  # 5-class: {0:Left, 1:Lean Left, 2:Center, 3:Lean Right, 4:Right}


def generate_recommendations(df, embeddings, top_k=2):
    """
    Generate bias-based recommendations per cluster (AllSides-L 5-class).

    Args:
        df: DataFrame with cluster, label, article_id columns
        embeddings: (N, dim) embedding array (same order as df)
        top_k: Number of recommended articles per bias class

    Returns:
        recommendations: List of recommendation results
    """
    cluster_groups = defaultdict(list)
    for idx, cluster_id in enumerate(df['cluster'].values):
        cluster_groups[cluster_id].append(idx)

    bias_values = sorted(BIAS_MAP.keys())

    if 'predicted_label' not in df.columns or df['predicted_label'].isna().all():
        print("  [Error] predicted_label column not found.")
        print("  Run scripts/add_predicted_labels.py → main.py first.")
        return []
    label_col = 'predicted_label'
    print("  Using predicted_label for recommendations")

    recommendations = []
    total_clusters = len(cluster_groups)

    for i, (cluster_id, indices) in enumerate(cluster_groups.items()):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Progress: {i+1}/{total_clusters} clusters")

        indices = np.array(indices)
        cluster_embeddings = embeddings[indices]
        sim_matrix = cosine_similarity(cluster_embeddings)

        labels = df.iloc[indices][label_col].values
        bias_indices = {bias: np.where(labels == bias)[0] for bias in bias_values}

        for local_idx in range(len(indices)):
            global_idx = int(indices[local_idx])
            article_label = int(labels[local_idx])
            article_id = int(df.iloc[global_idx]['article_id'])
            row = df.iloc[global_idx]

            rec = {
                'article_id': article_id,
                'title': row['title'],
                'cluster': int(cluster_id),
                'predicted_label': article_label,
                'gold_label': int(row['label']),
                'bias': BIAS_MAP.get(article_label, f'Class_{article_label}'),
            }

            for bias_val, bias_name in BIAS_MAP.items():
                key = f'similar_{bias_name.lower().replace(" ", "_")}'
                candidates = bias_indices[bias_val]
                candidates = candidates[candidates != local_idx]

                if len(candidates) == 0:
                    rec[key] = []
                    continue

                sims = sim_matrix[local_idx, candidates]
                valid_mask = sims < 1.0
                candidates = candidates[valid_mask]
                sims = sims[valid_mask]

                if len(candidates) == 0:
                    rec[key] = []
                    continue

                k = min(top_k, len(candidates))
                top_indices = candidates[np.argsort(sims)[-k:][::-1]]
                rec[key] = [
                    {
                        'article_id': int(df.iloc[int(indices[c])]['article_id']),
                        'title': df.iloc[int(indices[c])]['title'],
                        'similarity': round(float(sim_matrix[local_idx, c]), 4),
                    }
                    for c in top_indices
                ]

            recommendations.append(rec)

    return recommendations


def print_sample_results(recommendations, n_samples=3):
    print("\n" + "=" * 70)
    print("Sample Recommendations")
    print("=" * 70)
    for rec in recommendations[:n_samples]:
        print(f"\n[{rec['bias']}] {rec['title'][:60]}...")
        print(f"   article_id: {rec['article_id']}, cluster: {rec['cluster']}")
        for bias_name in BIAS_MAP.values():
            key = f'similar_{bias_name.lower().replace(" ", "_")}'
            articles = rec.get(key, [])
            print(f"   {bias_name}:")
            if not articles:
                print(f"      (no articles with this bias)")
            for a in articles:
                print(f"      [{a['similarity']:.4f}] (id={a['article_id']}) {a['title'][:50]}...")


def print_coverage_stats(recommendations, top_k):
    print("\n" + "=" * 70)
    print("Recommendation Coverage Stats")
    print("=" * 70)
    total = len(recommendations)
    for bias_name in BIAS_MAP.values():
        key = f'similar_{bias_name.lower().replace(" ", "_")}'
        has_rec = sum(1 for r in recommendations if len(r.get(key, [])) > 0)
        full_rec = sum(1 for r in recommendations if len(r.get(key, [])) >= top_k)
        print(f"  {bias_name:<12}: available {has_rec:,}/{total:,} "
              f"({has_rec/total*100:.1f}%), "
              f"Top-{top_k} complete {full_rec:,} ({full_rec/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Bias-aware similar article recommendation (AllSides-L)")
    parser.add_argument("--full", action="store_true",
                        help="Use the full AllSides-L clustering results instead of the test set")
    parser.add_argument("--model_name", type=str, default="all-MiniLM-L6-v2",
                        help="SBERT model used for clustering (must match main.py)")
    parser.add_argument("--top_k", type=int, default=2,
                        help="Number of recommended articles per bias class (default: 2)")
    args = parser.parse_args()

    suffix = '_full' if args.full else ''
    model_output_dir = os.path.join(
        UNIFIED_ROOT, 'outputs', 'news_clustering', args.model_name + suffix
    )
    noise_dir = os.path.join(model_output_dir, 'remove_noise')
    rec_dir = os.path.join(UNIFIED_ROOT, 'outputs', 'recommendations')

    # ── Load data ──────────────────────────────────────────────
    print("Loading data...")
    clean_csv = os.path.join(noise_dir, 'clusters_clean.csv')
    if not os.path.exists(clean_csv):
        print(f"Clean cluster CSV not found: {clean_csv}")
        print("Run main.py to generate clustering results first.")
        return

    df = pd.read_csv(clean_csv)
    if 'article_id' not in df.columns:
        df['article_id'] = df.index

    # Slice embeddings to match noise-filtered rows
    embeddings_full = np.load(os.path.join(model_output_dir, 'embeddings.npy'))
    original_df = pd.read_csv(os.path.join(model_output_dir, 'clusters.csv'))
    embeddings = embeddings_full[original_df['cluster'] != -1]

    print(f"Articles: {len(df):,}")
    print(f"Embeddings: {embeddings.shape}")
    print(f"Clusters: {df['cluster'].nunique():,}")
    print(f"Bias classes: 5 ({', '.join(BIAS_MAP.values())})")

    # ── Generate recommendations ───────────────────────────────
    print("\nGenerating recommendations per cluster...")
    recommendations = generate_recommendations(df, embeddings, top_k=args.top_k)
    if not recommendations:
        return

    # ── Save results ───────────────────────────────────────────
    result = {
        'metadata': {
            'total_articles': len(recommendations),
            'total_clusters': int(df['cluster'].nunique()),
            'top_k_per_bias': args.top_k,
            'similarity_metric': 'cosine',
            'bias_labels': {str(k): v for k, v in BIAS_MAP.items()},
        },
        'recommendations': recommendations,
    }

    os.makedirs(rec_dir, exist_ok=True)
    output_path = os.path.join(rec_dir, 'recommendations.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nRecommendations saved: {output_path}")
    print_sample_results(recommendations)
    print_coverage_stats(recommendations, args.top_k)


if __name__ == '__main__':
    main()
