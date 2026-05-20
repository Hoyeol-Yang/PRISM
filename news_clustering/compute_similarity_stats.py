"""
Comparison of recommendation similarity vs. overall dataset similarity (AllSides-L test set).

1. Average cosine similarity of recommended article pairs.
   → Averages the similarity values from the similar_* fields in recommendations.json.

2. Pairwise cosine similarity of all noise-filtered articles (baseline).
   → Uses clusters_clean.csv + embeddings.npy.

Usage:
    cd news_clustering
    python compute_similarity_stats.py

    # Use a different SBERT model's results
    python compute_similarity_stats.py --model_name all-mpnet-base-v2
"""

import argparse
import json
import os
import numpy as np
import pandas as pd

CLUSTERING_ROOT = os.path.dirname(os.path.abspath(__file__))
UNIFIED_ROOT = os.path.dirname(CLUSTERING_ROOT)

BIAS_MAP = {0: 'Left', 1: 'Lean Left', 2: 'Center', 3: 'Lean Right', 4: 'Right'}
BIAS_KEYS = [f'similar_{v.lower().replace(" ", "_")}' for v in BIAS_MAP.values()]


def main():
    parser = argparse.ArgumentParser(description='Recommendation similarity stats (AllSides-L test)')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--full', action='store_true',
                        help='Use clustering results generated with --full')
    args = parser.parse_args()

    suffix = '_full' if args.full else ''
    base = os.path.join(UNIFIED_ROOT, 'outputs', 'news_clustering',
                        args.model_name + suffix)
    noise_dir = os.path.join(base, 'remove_noise')

    recs_path = os.path.join(UNIFIED_ROOT, 'outputs', 'recommendations', 'recommendations.json')
    csv_path  = os.path.join(noise_dir, 'clusters_clean.csv')
    emb_path  = os.path.join(base, 'embeddings.npy')

    for p in [recs_path, csv_path, emb_path]:
        if not os.path.exists(p):
            print(f"File not found: {p}")
            print("Run main.py → recommend.py first.")
            return

    print("=" * 60)
    print("Loading data...")
    with open(recs_path) as f:
        data = json.load(f)
    recs = data['recommendations']

    emb_all = np.load(emb_path)
    df = pd.read_csv(csv_path, usecols=['article_id', 'cluster'])
    print(f"  Recommended articles: {len(recs):,}")
    print(f"  Embedding shape:      {emb_all.shape}")
    print(f"  Clean articles:       {len(df):,}")

    # ── 1. Recommended article pair similarity ─────────────────
    recommended_sims = []
    bias_sims = {b: [] for b in BIAS_MAP}
    n_rec = 0

    for rec in recs:
        sims = [s['similarity'] for k in BIAS_KEYS for s in rec.get(k, [])]
        if sims:
            recommended_sims.extend(sims)
            bias_sims[rec['predicted_label']].extend(sims)
            n_rec += 1

    avg_rec = np.mean(recommended_sims)
    std_rec = np.std(recommended_sims)

    print(f"\n[Recommended article pairs]")
    print(f"  Articles with recommendations: {n_rec:,}")
    print(f"  Total similarity pairs:        {len(recommended_sims):,}")
    print(f"  Avg similarity:                {avg_rec:.4f}  (std={std_rec:.4f})")
    print(f"  Min/Max:                       {min(recommended_sims):.4f} / {max(recommended_sims):.4f}")
    print(f"  Median:                        {np.median(recommended_sims):.4f}")

    print(f"\n  [By source article bias]")
    print(f"  {'Bias':<12} {'Mean':>10} {'Std':>8} {'Median':>10} {'Pairs':>10}")
    print(f"  {'-'*52}")
    for label_id, bias_name in BIAS_MAP.items():
        sims = bias_sims[label_id]
        if sims:
            print(f"  {bias_name:<12} {np.mean(sims):>10.4f} {np.std(sims):>8.4f} "
                  f"{np.median(sims):>10.4f} {len(sims):>10,}")
        else:
            print(f"  {bias_name:<12} {'N/A':>10}")

    # ── 2. Overall pairwise similarity (baseline) ─────────────
    print(f"\n[Overall pairwise similarity (baseline)]")
    all_ids = df['article_id'].tolist()
    n = len(all_ids)
    print(f"  Articles: {n:,}  |  Pairs: {n*(n-1)//2:,}")
    print("  Computing...")

    embs = emb_all[all_ids].astype(np.float32)

    BATCH = 500
    total_sum = 0.0
    total_count = 0
    for i in range(0, n, BATCH):
        end_i = min(i + BATCH, n)
        batch = embs[i:end_i]
        sim_block = batch @ embs.T
        for local_row, global_i in enumerate(range(i, end_i)):
            row_sims = sim_block[local_row, global_i + 1:]
            total_sum += float(row_sims.sum())
            total_count += len(row_sims)
        if (i // BATCH) % 10 == 0:
            print(f"  ... {min(end_i, n)}/{n}")

    avg_baseline = total_sum / total_count

    sum_sq = 0.0
    for i in range(0, n, BATCH):
        end_i = min(i + BATCH, n)
        batch = embs[i:end_i]
        sim_block = batch @ embs.T
        for local_row, global_i in enumerate(range(i, end_i)):
            row_sims = sim_block[local_row, global_i + 1:]
            sum_sq += float(((row_sims - avg_baseline) ** 2).sum())
    std_baseline = np.sqrt(sum_sq / total_count)

    print(f"  Total pairs:    {total_count:,}")
    print(f"  Avg similarity: {avg_baseline:.4f}  (std={std_baseline:.4f})")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Clean articles:             {n:,}")
    print(f"  Articles with recs:         {n_rec:,}")
    print(f"  [Recommended pairs] Avg:    {avg_rec:.4f}  (std={std_rec:.4f},  n={len(recommended_sims):,})")
    print(f"  [All pairs]         Avg:    {avg_baseline:.4f}  (std={std_baseline:.4f},  n={total_count:,})")
    print(f"  Difference (recs - all):    {avg_rec - avg_baseline:+.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
