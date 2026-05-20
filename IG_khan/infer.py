"""
KHAN inference module (AllSides-L only).

Classifies the political bias of news articles using a trained KHAN model
(allsides-l_best.pt).

Usage:
    python infer.py --title "..." --text "..."
    python infer.py --num_articles 5
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UNIFIED_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, UNIFIED_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from utils import load_vocab, load_knowledge_indices, load_model, preprocess_article, ALLSIDES_L_LABELS


class KHANPredictor:
    """KHAN bias classifier (AllSides-L 5-class)."""

    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Args:
            model_path: Checkpoint path (None → auto-detect under models/khan/)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.label_names = ALLSIDES_L_LABELS

        model_dir = os.path.join(UNIFIED_ROOT, 'models', 'khan')
        self.vocab, num_class = load_vocab(model_dir)

        pre_trained_dir = os.path.join(PROJECT_ROOT, 'pre-trained')
        entity_lists = load_knowledge_indices(pre_trained_dir)
        self.knowledge_indices = {
            k: self.vocab.lookup_indices(v) for k, v in entity_lists.items()
        }

        if model_path is None:
            model_dir = os.path.join(UNIFIED_ROOT, 'models', 'khan')
            pts = [f for f in os.listdir(model_dir) if f.endswith('.pt') and 'allsides-l' in f.lower()]
            if not pts:
                pts = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
            if not pts:
                raise FileNotFoundError(f"No .pt file found in models/khan/")
            model_path = os.path.join(model_dir, pts[0])

        self.model = load_model(
            model_path=model_path,
            vocab=self.vocab,
            num_class=num_class,
            knowledge_indices=self.knowledge_indices,
            device=device,
        )
        self.model.eval()

    def predict(self, title: str, text: str) -> Dict:
        """
        Predict the bias of a single article.

        Returns:
            {prediction, label, confidence, probabilities, logits}
        """
        titles_tensor, sentences_tensor, _ = preprocess_article(
            title, text, self.vocab, device=self.device
        )

        with torch.no_grad():
            output = self.model(sentences_tensor, titles_tensor, return_attention=False)

        probs = F.softmax(output, dim=-1)
        pred_label = output.argmax(dim=-1).item()
        confidence = probs[0, pred_label].item()

        prob_dict = {name: round(probs[0, idx].item(), 4) for idx, name in self.label_names.items()}
        logit_dict = {name: round(output[0, idx].item(), 4) for idx, name in self.label_names.items()}

        return {
            'prediction': self.label_names.get(pred_label, str(pred_label)),
            'label': pred_label,
            'confidence': round(confidence, 4),
            'probabilities': prob_dict,
            'logits': logit_dict,
        }

    def predict_batch(self, articles: List[Dict]) -> List[Dict]:
        """Predict bias for a batch of articles."""
        results = []
        for article in articles:
            result = self.predict(article['title'], article['text'])
            result['article_id'] = article.get('article_id')
            result['title'] = article['title']
            results.append(result)
        return results


def main():
    parser = argparse.ArgumentParser(description='KHAN bias classifier (AllSides-L 5-class)')
    parser.add_argument('--title', type=str, help='Article title (single inference)')
    parser.add_argument('--text', type=str, help='Article body (single inference)')
    parser.add_argument('--num_articles', type=int, default=5,
                        help='Number of articles to sample from AllSides-L test (default: 5)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print("\n[KHAN] Loading model...")
    predictor = KHANPredictor(device=args.device)
    print("  Model loaded")

    if args.title and args.text:
        result = predictor.predict(args.title, args.text)
        print(f"\nTitle: {args.title[:70]}")
        print(f"Prediction: {result['prediction']} ({result['confidence']:.1%})")
        print(f"Probabilities: {result['probabilities']}")
    else:
        import pandas as pd
        data_dir = os.path.join(UNIFIED_ROOT, 'data')
        df = pd.read_csv(os.path.join(data_dir, 'AllSides-L_test.csv'))
        df.dropna(subset=['title', 'text', 'label'], inplace=True)
        if 'article_id' not in df.columns:
            df['article_id'] = df.index

        sample = df.sample(n=min(args.num_articles, len(df)), random_state=42)
        articles = [
            {'article_id': int(row['article_id']), 'title': row['title'], 'text': row['text']}
            for _, row in sample.iterrows()
        ]

        results = predictor.predict_batch(articles)
        print(f"\nInference results for {len(results)} articles (AllSides-L test):")
        for r in results:
            print(f"  [{r['article_id']}] {r['prediction']:<12} ({r['confidence']:.1%}) | {r['title'][:55]}...")


if __name__ == '__main__':
    main()
