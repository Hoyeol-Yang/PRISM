"""
Common utilities for IG_khan.

Handles KHAN model loading, vocabulary construction, article preprocessing,
and knowledge graph index loading. Specific to AllSides-L — vocab_train.pkl based.
"""

import os
import sys
import pickle
import torch
from typing import List, Tuple, Dict
from collections import Counter
from torchtext.data.utils import get_tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import KHANModel


basic_english_tokenizer = get_tokenizer('basic_english')


# ============================================================
# Vocab
# ============================================================

class Vocab:
    """Simple vocabulary (torchtext Vocab compatible interface)."""

    def __init__(self, counter: Counter, specials=None):
        self.itos = []
        self.stoi = {}

        if specials:
            for token in specials:
                self.itos.append(token)
                self.stoi[token] = len(self.itos) - 1

        for word, _ in counter.most_common():
            if word not in self.stoi:
                self.itos.append(word)
                self.stoi[word] = len(self.itos) - 1

        self.default_index = 0

    def set_default_index(self, index: int) -> None:
        self.default_index = index

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    def __call__(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.default_index) for t in tokens]

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.default_index) for t in tokens]


def load_vocab(model_dir: str) -> Tuple['Vocab', int]:
    """Load the AllSides-L vocabulary from models/khan/."""
    vocab_path = os.path.join(model_dir, 'vocab_train.pkl')
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"vocab_train.pkl not found: {vocab_path}\n"
            f"  Place the vocab_train.pkl generated during KHAN training in models/khan/."
        )
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab, 5  # AllSides-L: 5-class


# ============================================================
# Knowledge Graph indices
# ============================================================

def load_knowledge_indices(pre_trained_dir: str) -> Dict[str, List[str]]:
    """Read KG entity dictionaries (con/lib/yago) and return token lists."""
    rep, demo, common = [], [], []

    with open(os.path.join(pre_trained_dir, 'entities_con.dict')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                rep.append(parts[1])

    with open(os.path.join(pre_trained_dir, 'entities_lib.dict')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                demo.append(parts[1])

    with open(os.path.join(pre_trained_dir, 'entities_yago.dict')) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                common.append(parts[1].split('_')[0].lower())

    return {'rep': rep, 'demo': demo, 'common': common}


# ============================================================
# Model loading
# ============================================================

def load_model(
    model_path: str,
    vocab: Vocab,
    num_class: int,
    knowledge_indices: Dict[str, List[int]],
    embed_size: int = 256,
    nhead: int = 4,
    d_hid: int = 128,
    nlayers: int = 1,
    dropout: float = 0.3,
    alpha: float = 0.6,
    beta: float = 0.2,
    device: str = 'cpu',
) -> KHANModel:
    """Load a trained KHAN checkpoint (model is built to match checkpoint dimensions)."""
    checkpoint = torch.load(model_path, map_location=device)

    ckpt_vocab_size = checkpoint['embeddings.weight'].shape[0]
    ckpt_embed_size = checkpoint['embeddings.weight'].shape[1]
    ckpt_num_class = checkpoint['classifier.weight'].shape[0]

    print(f"  Checkpoint: vocab_size={ckpt_vocab_size:,}, "
          f"embed_size={ckpt_embed_size}, num_class={ckpt_num_class}")

    adjusted_knowledge_indices = {
        k: [idx if idx < ckpt_vocab_size else 0 for idx in v]
        for k, v in knowledge_indices.items()
    }

    model = KHANModel(
        vocab_size=ckpt_vocab_size,
        embed_size=ckpt_embed_size,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        dropout=dropout,
        num_class=ckpt_num_class,
        knowledge_indices=adjusted_knowledge_indices,
        alpha=alpha,
        beta=beta,
    )
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    model.checkpoint_vocab_size = ckpt_vocab_size
    model.checkpoint_num_class = ckpt_num_class
    return model


# ============================================================
# Article preprocessing
# ============================================================

def preprocess_article(
    title: str,
    text: str,
    vocab: Vocab,
    max_sentence: int = 40,
    max_words: int = 40,
    device: str = 'cpu',
) -> Tuple[torch.Tensor, torch.Tensor, List[List[List[str]]]]:
    """
    Preprocess an article into KHAN model input format.

    Returns:
        titles_tensor: [1, max_words]
        sentences_tensor: [1, max_sentence, max_words]
        tokens: [batch][sent_idx][word_idx] (for visualization / IG score mapping)
    """
    title_tokens = basic_english_tokenizer(str(title))
    title_indices = vocab(title_tokens)

    if len(title_indices) < max_words:
        title_indices.extend([0] * (max_words - len(title_indices)))
    else:
        title_indices = title_indices[:max_words]

    text_tokens = basic_english_tokenizer(text)
    text_indices = vocab(text_tokens)

    sentences_list, tokens_list = [], []
    cur_sent, cur_toks = [], []

    for idx, token_idx in enumerate(text_indices):
        if token_idx == 1:  # <sep>
            if cur_sent:
                sentences_list.append(cur_sent)
                tokens_list.append(cur_toks)
                cur_sent, cur_toks = [], []
        else:
            cur_sent.append(token_idx)
            if idx < len(text_tokens):
                cur_toks.append(text_tokens[idx])

    if cur_sent:
        sentences_list.append(cur_sent)
        tokens_list.append(cur_toks)

    processed_sentences, processed_tokens = [], []
    for i, (sent, toks) in enumerate(zip(sentences_list, tokens_list)):
        if i >= max_sentence:
            break
        if len(sent) < max_words:
            processed_sentences.append(sent + [0] * (max_words - len(sent)))
            processed_tokens.append(toks + ['<pad>'] * (max_words - len(toks)))
        else:
            processed_sentences.append(sent[:max_words])
            processed_tokens.append(toks[:max_words])

    while len(processed_sentences) < max_sentence:
        processed_sentences.append([0] * max_words)
        processed_tokens.append(['<pad>'] * max_words)

    titles_tensor = torch.tensor([title_indices], dtype=torch.int64).to(device)
    sentences_tensor = torch.tensor([processed_sentences], dtype=torch.int64).to(device)

    return titles_tensor, sentences_tensor, [processed_tokens]


# ============================================================
# Labels
# ============================================================

ALLSIDES_L_LABELS = {
    0: 'Left',
    1: 'Lean Left',
    2: 'Center',
    3: 'Lean Right',
    4: 'Right',
}


def get_label_name(label_idx: int) -> str:
    """Return the AllSides-L 5-class label name for a given index."""
    return ALLSIDES_L_LABELS.get(label_idx, f'Class {label_idx}')
