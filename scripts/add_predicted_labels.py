"""
Script to add a predicted_label column to AllSides-L_test.csv.

Uses the KHAN model located in models/khan/ to predict the political bias
of each article and stores the result in the predicted_label column.

Usage:
    python scripts/add_predicted_labels.py
    python scripts/add_predicted_labels.py --device cpu
"""

import os
import sys
import time
import argparse
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UNIFIED_ROOT = os.path.dirname(SCRIPT_DIR)
KHAN_ROOT = os.path.join(UNIFIED_ROOT, 'IG_khan')
DATA_DIR = os.path.join(UNIFIED_ROOT, 'data')
sys.path.insert(0, UNIFIED_ROOT)
sys.path.insert(0, KHAN_ROOT)

from infer import KHANPredictor


def main():
    parser = argparse.ArgumentParser(description='Add predicted_label to AllSides-L_test.csv')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--csv', type=str,
                        default=os.path.join(DATA_DIR, 'AllSides-L_test.csv'),
                        help='Input CSV path')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(UNIFIED_ROOT, 'models', 'khan', 'allsides-l_best.pt'),
                        help='KHAN model path')
    parser.add_argument('--batch_log', type=int, default=500,
                        help='Progress log interval')
    args = parser.parse_args()

    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    df.dropna(subset=['title', 'text', 'label'], inplace=True)
    print(f"  - Articles: {len(df):,}")

    if 'predicted_label' in df.columns:
        existing = df['predicted_label'].notna().sum()
        print(f"  predicted_label already exists ({existing:,} entries)")
        overwrite = input("  Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("  Cancelled.")
            return

    print(f"\nLoading model...")
    print(f"  - Model: {args.model_path}")
    print(f"  - Device: {args.device}")
    predictor = KHANPredictor(
        dataset='ALLSIDES-L',
        model_path=args.model_path,
        device=args.device
    )
    print("  Model loaded")

    print(f"\nStarting inference ({len(df):,} articles)...")
    predicted_labels = []
    start_time = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        if i == 0 or (i + 1) % args.batch_log == 0:
            elapsed = time.time() - start_time
            speed = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(df) - i - 1) / speed if speed > 0 else 0
            print(f"  Progress: {i+1:,}/{len(df):,} "
                  f"({(i+1)/len(df)*100:.1f}%) "
                  f"| Speed: {speed:.1f} articles/s "
                  f"| ETA: {remaining/60:.0f}min")

        result = predictor.predict(row['title'], row['text'])
        predicted_labels.append(result['label'])

    df['predicted_label'] = predicted_labels

    accuracy = (df['predicted_label'] == df['label']).mean()
    elapsed = time.time() - start_time
    print(f"\nResults:")
    print(f"  - Elapsed: {elapsed/60:.1f}min")
    print(f"  - Gold label match rate: {accuracy:.1%}")

    df.to_csv(args.csv, index=False)
    print(f"\nSaved: {args.csv}")
    print(f"   Columns: {list(df.columns)}")


if __name__ == '__main__':
    main()
