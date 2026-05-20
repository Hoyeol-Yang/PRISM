"""
News clustering pipeline (AllSides-L only).

SBERT embeddings → HDBSCAN clustering → save results.
Uses a shared article_id scheme for interoperability with IG_khan.

Usage:
    # Default: AllSides-L test set
    python main.py

    # Full AllSides-L dataset
    python main.py --full

    # Change embedding model
    python main.py --model-name all-mpnet-base-v2

    # Force recompute embeddings
    python main.py --force-embed
"""

import os
import sys
import argparse
import pandas as pd

CLUSTERING_ROOT = os.path.dirname(os.path.abspath(__file__))
UNIFIED_ROOT = os.path.dirname(CLUSTERING_ROOT)
sys.path.insert(0, UNIFIED_ROOT)

from src.embeddings import create_embeddings, save_embeddings, load_embeddings
from src.clustering import cluster_articles, get_cluster_stats
from src.evaluate import (
    print_cluster_samples,
    get_clustering_summary,
    save_clusters_to_json,
    plot_cluster_sizes,
    plot_bias_distribution
)


def load_data(data_dir: str, full: bool) -> tuple:
    """Load AllSides-L data. Uses test set CSV when full=False."""
    if full:
        csv_path = os.path.join(data_dir, 'AllSides-L.csv')
        source_label = 'full'
    else:
        csv_path = os.path.join(data_dir, 'AllSides-L_test.csv')
        source_label = 'test set'

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.dropna(subset=['title', 'text', 'label'], inplace=True)
    if 'article_id' not in df.columns:
        df['article_id'] = df.index

    return df, source_label


def main(args):
    print("=" * 60)
    print("News Clustering Pipeline — AllSides-L")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────
    print("\n[1/4] Data preprocessing...")
    data_dir = os.path.join(UNIFIED_ROOT, 'data')
    df, source_label = load_data(data_dir, args.full)

    # Preserve raw text for KHAN inference; generate clustering text with <SEP> replaced
    df['_raw_text'] = df['text'].fillna('')
    df['text'] = df['_raw_text'].str.replace('<SEP>', ' ', regex=False)
    df['title'] = df['title'].fillna('')
    df['combined_text'] = (
        df['title'] + ' ' + df['text']
    ).str.replace(r'\s+', ' ', regex=True).str.strip()

    print(f"  - Dataset: AllSides-L ({source_label})")
    print(f"  - Total articles: {len(df):,}")

    # ── 2. Embeddings ─────────────────────────────────────────
    print("\n[2/4] Generating embeddings...")
    print(f"  - Model: {args.model_name}")

    suffix = '_full' if args.full else ''
    model_output_dir = os.path.join(
        UNIFIED_ROOT, 'outputs', 'news_clustering', args.model_name + suffix
    )
    os.makedirs(model_output_dir, exist_ok=True)
    print(f"  - Output folder: {model_output_dir}")

    embeddings_path = os.path.join(model_output_dir, 'embeddings.npy')
    if os.path.exists(embeddings_path) and not args.force_embed:
        print(f"  - Loading existing embeddings")
        embeddings = load_embeddings(embeddings_path)
    else:
        embeddings = create_embeddings(
            df['combined_text'].tolist(),
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
        save_embeddings(embeddings, embeddings_path)

    # ── 3. Clustering ─────────────────────────────────────────
    print("\n[3/4] Clustering...")
    cluster_labels, _ = cluster_articles(
        embeddings,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
    )
    df['cluster'] = cluster_labels

    # Check for predicted_label (not required for clustering — used by recommend.py)
    if 'predicted_label' in df.columns and df['predicted_label'].notna().any():
        n_pred = df['predicted_label'].notna().sum()
        accuracy = (df['predicted_label'] == df['label']).mean()
        print(f"\n[3.5] predicted_label found ({n_pred:,} articles, gold match rate: {accuracy:.1%})")
    else:
        print("\n[3.5] predicted_label not found — clustering continues")
        print("  Run scripts/add_predicted_labels.py before running recommend.py.")

    # ── 4. Save and evaluate ──────────────────────────────────
    print("\n[4/4] Saving results and evaluating...")
    df = df.drop(columns=['_raw_text'], errors='ignore')

    df.to_csv(os.path.join(model_output_dir, 'clusters.csv'), index=False)
    print(f"  - clusters.csv saved")

    noise_dir = os.path.join(model_output_dir, 'remove_noise')
    os.makedirs(noise_dir, exist_ok=True)
    df[df['cluster'] != -1].to_csv(os.path.join(noise_dir, 'clusters_clean.csv'), index=False)
    print(f"  - clusters_clean.csv saved")

    save_clusters_to_json(df, os.path.join(model_output_dir, 'clusters.json'))

    summary = get_clustering_summary(df)
    print("\n" + "=" * 60)
    print("Clustering Summary")
    print("=" * 60)
    print(f"  - Total articles:        {summary['total_articles']:,}")
    print(f"  - Clusters:              {summary['n_clusters']:,}")
    print(f"  - Noise articles:        {summary['n_noise']:,} ({summary['noise_ratio']*100:.1f}%)")
    print(f"  - Avg cluster size:      {summary['avg_cluster_size']:.1f}")
    print(f"  - Max:                   {summary['max_cluster_size']}")
    print(f"  - Min:                   {summary['min_cluster_size']}")

    if not args.no_samples:
        print()
        print_cluster_samples(df, n_samples=3, n_clusters=10)

    plot_cluster_sizes(df, save_path=os.path.join(model_output_dir, 'cluster_sizes.png'))
    plot_bias_distribution(df, save_path=os.path.join(model_output_dir, 'bias_distribution.png'))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News Clustering Pipeline (AllSides-L)")

    parser.add_argument("--full", action="store_true",
                        help="Use the full AllSides-L dataset instead of the test set")

    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2",
                        help="SBERT model (default: all-MiniLM-L6-v2)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--force-embed", action="store_true",
                        help="Recompute embeddings even if a cached version exists")

    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--no-samples", action="store_true",
                        help="Disable cluster sample output")

    args = parser.parse_args()
    main(args)
