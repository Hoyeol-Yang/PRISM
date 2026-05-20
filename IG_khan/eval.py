"""
KHAN model evaluation (AllSides-L only).

Reports both AllSides-L 5-class strict accuracy and 3-class group-mapped accuracy.

Test set split:
    Reproduced using the same method as khan-www23/data_utils.py:
    StratifiedKFold(n_splits=10, shuffle=True, random_state=10), 1st fold test set.

Usage:
    # Evaluate on AllSides-L test set (default)
    python eval.py

    # Evaluate on full dataset
    python eval.py --full
"""

import sys
import os
import argparse
import time
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UNIFIED_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, UNIFIED_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from utils import load_vocab, load_knowledge_indices, load_model, preprocess_article


LABEL_5 = {0: 'Left', 1: 'Lean Left', 2: 'Center', 3: 'Lean Right', 4: 'Right'}
LABEL_3 = {0: 'Left', 1: 'Center', 2: 'Right'}


def map_5to3(label_5: int) -> int:
    """5-class → grouped 3-class (Left+LeanLeft→0, Center→1, LeanRight+Right→2)."""
    if label_5 in (0, 1):
        return 0
    elif label_5 == 2:
        return 1
    return 2


def get_test_indices(data_dir: str):
    """Return AllSides-L test indices using the same split as khan-www23/data_utils.py."""
    csv_path = os.path.join(data_dir, 'AllSides-L.csv')
    dataset = pd.read_csv(csv_path)
    dataset.dropna(inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=10)
    for train_idx, test_idx in skf.split(dataset[['text', 'title']], dataset[['label']]):
        return train_idx, test_idx


def print_confusion_matrix(matrix: np.ndarray, label_names: dict, title: str):
    n = len(label_names)
    print(f"  [{title}]")
    print(f"  {'':>14} ", end='')
    for c in range(n):
        print(f"{'P-' + label_names[c]:>12}", end='')
    print()
    for r in range(n):
        print(f"  {'T-' + label_names[r]:>14} ", end='')
        for c in range(n):
            print(f"{matrix[r, c]:>12}", end='')
        print()
    print()


def print_class_accuracy(matrix: np.ndarray, label_names: dict, title: str):
    n = len(label_names)
    print(f"  [{title}]")
    print(f"  {'Class':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("  " + "-" * 44)
    for c in range(n):
        total = int(matrix[c].sum())
        correct = int(matrix[c, c])
        acc = correct / total if total > 0 else 0.0
        print(f"  {label_names[c]:<14} {correct:>8} {total:>8} {acc:>10.4f}")
    print()


def run_eval(df: pd.DataFrame, model, vocab, device: str):
    correct_strict = 0
    correct_group = 0
    total = 0
    errors = 0

    confusion_5 = np.zeros((5, 5), dtype=int)
    confusion_3 = np.zeros((3, 3), dtype=int)

    start = time.time()
    n_rows = len(df)

    for i, (_, row) in enumerate(df.iterrows()):
        true_5 = int(row['label'])
        true_3 = map_5to3(true_5)

        try:
            t_t, s_t, _ = preprocess_article(
                str(row['title']), str(row['text']), vocab, device=device
            )
            with torch.no_grad():
                output = model(s_t, t_t, return_attention=False)

            pred_5 = output.argmax(dim=-1).item()
            pred_3 = map_5to3(pred_5)

            if pred_5 == true_5:
                correct_strict += 1
            if pred_3 == true_3:
                correct_group += 1

            confusion_5[true_5, pred_5] += 1
            confusion_3[true_3, pred_3] += 1
            total += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [Error] article_id={row.get('article_id', '?')}: {e}")

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            eta = elapsed / (i + 1) * (n_rows - i - 1)
            print(f"  {i+1:>6}/{n_rows} ({100*(i+1)/n_rows:.1f}%) | "
                  f"strict={correct_strict/total:.4f} | "
                  f"group={correct_group/total:.4f} | ETA={eta:.0f}s")

    elapsed = time.time() - start
    print(f"\n{'=' * 65}")
    print(f"  AllSides-L ({n_rows:,} articles) | succeeded: {total:,} | errors: {errors}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 65}\n")

    # Strict 5-class
    acc_strict = correct_strict / total
    print(f"  * Strict 5-class accuracy: {acc_strict:.4f} ({correct_strict}/{total})")
    print()
    print_class_accuracy(confusion_5, LABEL_5, "Per-class accuracy — 5-class")
    print_confusion_matrix(confusion_5, LABEL_5, "Confusion Matrix — 5-class")

    # Group 3-class
    acc_group = correct_group / total
    print(f"  * Group 3-class accuracy: {acc_group:.4f} ({correct_group}/{total})")
    print(f"    (Left+Lean Left→Left, Center→Center, Lean Right+Right→Right)")
    print()
    print_class_accuracy(confusion_3, LABEL_3, "Per-class accuracy — Group 3-class")
    print_confusion_matrix(confusion_3, LABEL_3, "Confusion Matrix — Group 3-class")


def main():
    parser = argparse.ArgumentParser(description='KHAN evaluation (AllSides-L)')
    parser.add_argument('--full', action='store_true',
                        help='Evaluate on the full dataset instead of the test set')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"Device: {args.device}")
    data_dir = os.path.join(UNIFIED_ROOT, 'data')
    model_path = os.path.join(UNIFIED_ROOT, 'models', 'khan', 'allsides-l_best.pt')

    print("\n[1/2] Loading data and model...")
    full_df = pd.read_csv(os.path.join(data_dir, 'AllSides-L.csv'))
    full_df.dropna(inplace=True)
    full_df.reset_index(drop=True, inplace=True)

    if args.full:
        df = full_df
        print(f"  AllSides-L full: {len(df):,} articles")
    else:
        _, test_idx = get_test_indices(data_dir)
        df = full_df.iloc[test_idx].copy()
        print(f"  AllSides-L full: {len(full_df):,} → test set: {len(df):,} "
              f"(StratifiedKFold 10-fold, 1st fold)")
    print(f"  Label distribution: {df['label'].value_counts().sort_index().to_dict()}")

    model_dir = os.path.join(UNIFIED_ROOT, 'models', 'khan')
    vocab, num_class = load_vocab(model_dir)
    print(f"  Vocab size: {len(vocab):,}")

    entity_lists = load_knowledge_indices(os.path.join(PROJECT_ROOT, 'pre-trained'))
    knowledge_indices = {k: vocab.lookup_indices(v) for k, v in entity_lists.items()}

    model = load_model(
        model_path=model_path,
        vocab=vocab,
        num_class=num_class,
        knowledge_indices=knowledge_indices,
        device=args.device,
    )
    model.eval()
    print("  Model loaded\n")

    print("[2/2] Evaluating...")
    run_eval(df, model, vocab, args.device)


if __name__ == '__main__':
    main()
