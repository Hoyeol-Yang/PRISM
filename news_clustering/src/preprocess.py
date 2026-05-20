"""
Preprocessing module.
- Text cleaning (<SEP> token handling, etc.)
- Title + body combination.

Note: Data loading is handled by common.data_loader.
This module only handles text preprocessing after a DataFrame has been loaded.
"""

import pandas as pd
from typing import Optional


def preprocess_df(df: pd.DataFrame,
                  combine_title: bool = True,
                  max_text_length: Optional[int] = None) -> pd.DataFrame:
    """
    Preprocess text in a DataFrame.

    Args:
        df: DataFrame with article_id, title, text, label columns
        combine_title: Whether to concatenate title and body
        max_text_length: Maximum text length in characters (None = no limit)

    Returns:
        Preprocessed DataFrame (article_id preserved)
    """
    df = df.copy()

    df['text'] = df['text'].fillna('').str.replace('<SEP>', ' ', regex=False)
    df['title'] = df['title'].fillna('')

    if combine_title:
        df['combined_text'] = df['title'] + ' ' + df['text']
    else:
        df['combined_text'] = df['text']

    df['combined_text'] = df['combined_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    if max_text_length:
        df['combined_text'] = df['combined_text'].str[:max_text_length]

    return df


def get_label_name(label: int) -> str:
    """Convert an AllSides-L 5-class label index to its name."""
    label_map = {0: 'Left', 1: 'Lean Left', 2: 'Center', 3: 'Lean Right', 4: 'Right'}
    return label_map.get(label, 'Unknown')


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data'
    )
    df = pd.read_csv(os.path.join(data_dir, 'AllSides-L_test.csv'))
    df.dropna(subset=['title', 'text', 'label'], inplace=True)
    if 'article_id' not in df.columns:
        df['article_id'] = df.index

    df = preprocess_df(df)
    print(f"Total articles: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"article_id range: {df['article_id'].min()} ~ {df['article_id'].max()}")
    print(f"\nFirst article (first 200 chars):")
    print(df['combined_text'].iloc[0][:200])
