"""
Common Data Loader
==================

Both projects (IG_khan, news_clustering) use the same article_id scheme.

article_id assignment rules:
    - The row index (0-indexed) in the CSV is used as the article_id.
    - Assigned before dropna so the original row number is preserved after NaN removal.
    - Both projects use the same CSV, so the same article maps to the same article_id.

Usage:
    from common.data_loader import load_dataset
    df = load_dataset('ALLSIDES-S')
    # Articles can be identified by df['article_id']
"""

import os
import pandas as pd
from typing import Optional

# Data directory relative to the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, 'data')

DATASETS = {
    'ALLSIDES-L': {'file': 'AllSides-L.csv', 'num_class': 5},
    'ALLSIDES-S': {'file': 'AllSides-S.csv', 'num_class': 3},
    'SEMEVAL':    {'file': 'SemEval.csv',     'num_class': 2},
}

BIAS_MAP = {0: 'Left', 1: 'Center', 2: 'Right'}
BIAS_MAP_SEMEVAL = {0: 'Left', 1: 'Right'}
BIAS_MAP_ALLSIDES_L = {0: 'Left', 1: 'Lean Left', 2: 'Center', 3: 'Lean Right', 4: 'Right'}


def load_dataset(
    dataset: str = 'ALLSIDES-S',
    data_dir: Optional[str] = None,
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Load a dataset and assign article_id.

    Args:
        dataset: Dataset name ('ALLSIDES-S', 'SEMEVAL', etc.)
        data_dir: Data directory (default: project root/data)
        drop_na: Whether to drop rows with NaN values

    Returns:
        DataFrame with an article_id column
    """
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASETS.keys())}")

    if data_dir is None:
        data_dir = DATA_DIR

    info = DATASETS[dataset]
    csv_path = os.path.join(data_dir, info['file'])

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # article_id = original CSV row index (assigned before dropna)
    df['article_id'] = df.index

    if drop_na:
        df.dropna(subset=['title', 'text', 'label'], inplace=True)

    return df


def get_num_class(dataset: str) -> int:
    """Return the number of classes for the dataset."""
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DATASETS[dataset]['num_class']


def get_bias_map(dataset: str) -> dict:
    """Return the bias label mapping for the dataset."""
    if dataset == 'SEMEVAL':
        return BIAS_MAP_SEMEVAL
    elif dataset == 'ALLSIDES-L':
        return BIAS_MAP_ALLSIDES_L
    return BIAS_MAP


if __name__ == '__main__':
    # quick sanity check
    df = load_dataset('ALLSIDES-S')
    print(f"Total articles: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"article_id range: {df['article_id'].min()} ~ {df['article_id'].max()}")
    print(f"\nFirst 3 articles:")
    for _, row in df.head(3).iterrows():
        print(f"  [{row['article_id']}] {row['title'][:60]}...")
