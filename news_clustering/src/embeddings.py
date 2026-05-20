"""
Embeddings module.
- Text embedding generation using SBERT models.
- Embedding save/load utilities.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Optional
from tqdm import tqdm
import os


def create_embeddings(texts: List[str],
                      model_name: str = "all-MiniLM-L6-v2",
                      batch_size: int = 32,
                      show_progress: bool = True) -> np.ndarray:
    """
    Generate text embeddings using SBERT.

    Args:
        texts: List of input texts
        model_name: SBERT model name to use
        batch_size: Batch size
        show_progress: Whether to display a progress bar

    Returns:
        Embedding array (N x embedding_dim)
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Generating embeddings... ({len(texts)} texts)")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )

    print(f"Embeddings complete: shape = {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings: np.ndarray, path: str) -> None:
    """Save embeddings to a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, embeddings)
    print(f"Embeddings saved: {path}")


def load_embeddings(path: str) -> np.ndarray:
    """Load saved embeddings."""
    embeddings = np.load(path)
    print(f"Embeddings loaded: {path}, shape = {embeddings.shape}")
    return embeddings


if __name__ == "__main__":
    # Test with sample texts
    sample_texts = [
        "The president announced new economic policies today.",
        "Scientists discover new species in the Amazon rainforest.",
        "Tech company releases latest smartphone model.",
    ]

    embeddings = create_embeddings(sample_texts)
    print(f"Test embedding shape: {embeddings.shape}")
