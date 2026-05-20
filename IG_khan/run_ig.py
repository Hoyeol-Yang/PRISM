"""
IG_khan — Integrated Gradients XAI pipeline (AllSides-L only).

Classifies news articles from the AllSides-L test set using the KHAN model,
then extracts key bias-predicting keywords via Integrated Gradients.

Usage:
    # Default: analyze 5 articles from the AllSides-L test set
    python run_ig.py

    # Specify number of articles
    python run_ig.py --num_articles 100

    # Analyze specific article IDs
    python run_ig.py --article_ids 1000,2000,3000

    # Run without spaCy keyword expansion
    python run_ig.py --no_expand

    # Verbose output
    python run_ig.py --verbose
"""

import sys
import os
import random
import argparse
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UNIFIED_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, UNIFIED_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from utils import load_vocab, load_knowledge_indices, load_model, preprocess_article, ALLSIDES_L_LABELS
from ig_analyzer import IntegratedGradientsAnalyzer
from keyword_expander import KeywordExpander, XAIJsonExporter


def parse_args():
    parser = argparse.ArgumentParser(
        description='IG_khan — Integrated Gradients XAI (AllSides-L)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--all', action='store_true',
                        help='Analyze all articles in the test set (overrides --num_articles)')
    parser.add_argument('--num_articles', type=int, default=5,
                        help='Number of articles to analyze (default: 5, ignored with --all)')
    parser.add_argument('--article_ids', type=str, default=None,
                        help='Specific article IDs to analyze (comma-separated, e.g., 1000,2000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of keywords to extract (default: 10)')
    parser.add_argument('--ig_steps', type=int, default=50,
                        help='Number of IG interpolation steps (default: 50)')
    parser.add_argument('--no_expand', action='store_true',
                        help='Disable spaCy keyword expansion')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: {project_root}/outputs/IG_khan/)')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def select_articles(df, num_articles, article_ids, seed, use_all=False):
    if article_ids:
        ids = [int(x.strip()) for x in article_ids.split(',')]
        df = df[df['article_id'].isin(ids)]
        print(f"   Selected article_ids: {ids}")
    elif use_all:
        print(f"   Using all articles: {len(df):,}")
    elif num_articles < len(df):
        # Stratified sampling by class
        random.seed(seed)
        num_classes = df['label'].nunique()
        per_class = max(1, num_articles // num_classes)
        sampled = []
        for label in sorted(df['label'].unique()):
            cls_df = df[df['label'] == label]
            sampled.extend(cls_df.sample(n=min(per_class, len(cls_df)),
                                         random_state=seed).index.tolist())
        df = df.loc[sampled[:num_articles]]
        print(f"   Sampled: {len(df)} articles (stratified by class)")
    else:
        print(f"   Using all articles: {len(df):,}")

    return [
        {
            'article_id': int(row['article_id']),
            'title': row['title'],
            'text': row['text'],
            'true_label': int(row['label']),
            'true_label_name': ALLSIDES_L_LABELS.get(int(row['label']), '?'),
        }
        for _, row in df.iterrows()
    ]


def split_sentences(text: str):
    return [s.strip() for s in str(text).split('<SEP>') if s.strip()]


def main():
    args = parse_args()

    output_dir = args.output_dir or os.path.join(UNIFIED_ROOT, 'outputs', 'IG_khan')

    article_desc = "all" if args.all else str(args.num_articles)
    print("\n" + "=" * 65)
    print("  IG_khan — Integrated Gradients XAI")
    print("  Dataset: AllSides-L test set")
    print(f"  Articles: {article_desc} | IG steps: {args.ig_steps}")
    print("=" * 65)

    # --- Load data ---
    data_dir = os.path.join(UNIFIED_ROOT, 'data')
    test_csv = os.path.join(data_dir, 'AllSides-L_test.csv')
    if not os.path.exists(test_csv):
        print(f"AllSides-L_test.csv not found: {test_csv}")
        return

    df = pd.read_csv(test_csv)
    df.dropna(subset=['title', 'text', 'label'], inplace=True)
    if 'article_id' not in df.columns:
        df['article_id'] = df.index
    print(f"\nData loaded: {len(df):,} articles")

    articles = select_articles(df, args.num_articles, args.article_ids, args.seed, use_all=args.all)
    if not articles:
        print("No articles selected.")
        return

    # --- Load model ---
    print("\nLoading model...")
    model_dir = os.path.join(UNIFIED_ROOT, 'models', 'khan')
    vocab, num_class = load_vocab(model_dir)

    pre_trained_dir = os.path.join(PROJECT_ROOT, 'pre-trained')
    entity_lists = load_knowledge_indices(pre_trained_dir)
    knowledge_indices = {k: vocab.lookup_indices(v) for k, v in entity_lists.items()}

    model_dir = os.path.join(UNIFIED_ROOT, 'models', 'khan')
    pts = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'allsides-l' in f.lower()]
    if not pts:
        pts = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not pts:
        print("No .pt file found in models/khan/")
        return
    model_path = os.path.join(model_dir, pts[0])

    model = load_model(
        model_path=model_path,
        vocab=vocab,
        num_class=num_class,
        knowledge_indices=knowledge_indices,
        device=args.device,
    )
    print(f"  Model: {pts[0]}")
    print(f"  Vocab: {len(vocab):,} tokens")

    # --- IG analyzer / keyword expander / JSON exporter ---
    analyzer = IntegratedGradientsAnalyzer(model=model, device=args.device, steps=args.ig_steps)
    expander = None if args.no_expand else KeywordExpander(model='en_core_web_lg')
    exporter = XAIJsonExporter(output_dir=output_dir)

    # --- Run analysis ---
    correct = 0
    total_keywords = 0
    total_expanded = 0

    for i, article in enumerate(articles, 1):
        print(f"\n{'─' * 65}")
        print(f"  [{i}/{len(articles)}] {article['title'][:55]}...")
        print(f"     True label: {article['true_label_name']}")

        sentences = split_sentences(article['text'])

        try:
            titles_tensor, sentences_tensor, tokens = preprocess_article(
                article['title'], article['text'], vocab, device=args.device
            )

            result = analyzer.analyze(
                sentences_tensor=sentences_tensor,
                titles_tensor=titles_tensor,
                tokens=tokens,
                sentences=sentences,
                label_names=ALLSIDES_L_LABELS,
                top_k_words=args.top_k,
            )

            is_correct = (result.predicted_label == article['true_label'])
            correct += int(is_correct)
            total_keywords += len(result.top_words)

            status = "O" if is_correct else "X"
            print(f"     [{status}] Prediction: {result.predicted_label_name} ({result.confidence:.1%})")

            if args.verbose:
                print(f"     Keywords: {[w.word for w in result.top_words[:5]]}")

            # Compute probabilities
            with torch.no_grad():
                output = model(sentences_tensor, titles_tensor, return_attention=False)
                probs = F.softmax(output, dim=-1)
            probabilities = {name: round(probs[0, idx].item(), 4)
                             for idx, name in ALLSIDES_L_LABELS.items()}
            logits = {name: round(output[0, idx].item(), 4)
                      for idx, name in ALLSIDES_L_LABELS.items()}

            # Convert IG results to keyword_expander format
            raw_keywords = [
                {'word': w.word, 'gradient': w.score,
                 'attention': 0.0, 'combined': w.score}
                for w in result.top_words
            ]

            # Keyword expansion
            expanded_keywords = []
            if expander and raw_keywords:
                expanded_keywords = expander.expand_keywords(
                    article['text'], raw_keywords, top_k=args.top_k
                )
                expanded_count = sum(1 for kw in expanded_keywords if kw.original != kw.expanded)
                total_expanded += expanded_count
                if args.verbose and expanded_count > 0:
                    print(f"     Expanded: {expanded_count}")

            export_probs = {**probabilities, 'logits': logits}
            exporter.export_and_save(
                title=article['title'],
                text=article['text'],
                prediction_label=result.predicted_label_name,
                confidence=result.confidence,
                expanded_keywords=expanded_keywords,
                filename=f"article_{article['article_id']}",
                dataset='ALLSIDES-L',
                article_id=article['article_id'],
                probabilities=export_probs,
            )

        except Exception as e:
            print(f"     Error: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    # --- Summary ---
    n = len(articles)
    print("\n" + "=" * 65)
    print("  Analysis complete")
    print(f"  Articles: {n} | Accuracy: {correct}/{n} ({correct/n*100:.1f}%)")
    print(f"  Keywords extracted: {total_keywords}")
    if not args.no_expand:
        print(f"  Keywords expanded: {total_expanded}")
    print(f"  Output path: {output_dir}/")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    main()
