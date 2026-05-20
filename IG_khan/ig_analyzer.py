"""
Integrated Gradients analyzer for the KHAN model (self-contained).

Targets the output of model.knowledge_encoder (KnowledgeEncoding),
which is the fusion point of YAGO + KG-liberal + KG-conservative.
Political knowledge — the core differentiator of KHAN — is reflected
in the attribution scores.

Implementation notes:
    The knowledge_encoder inside KHAN's forward() sits inside a Python
    for-loop, making it incompatible with Captum LayerIntegratedGradients.
    IG is implemented manually using forward hooks to capture/replace outputs:
      1. Capture KE outputs for the baseline (zero token) and actual input.
      2. Interpolate between the two outputs (n_steps steps).
      3. Compute gradients at each interpolation point.
      4. Average gradients via the trapezoidal rule.
      5. IG = (actual - baseline) * avg_grad → L2 norm → min-max normalization.

IG axioms (Sundararajan et al., 2017):
    Sensitivity, Implementation Invariance, Completeness.

Reference:
    Sundararajan, M., Taly, A., & Yan, Q. (2017).
    Axiomatic Attribution for Deep Networks. ICML.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field


# ============================================================
# Result dataclasses
# ============================================================

@dataclass
class WordIG:
    """IG analysis result for a single word."""
    word: str
    score: float
    sentence_idx: int
    position: int


@dataclass
class SentenceIG:
    """IG analysis result for a single sentence."""
    sentence_idx: int
    text: str
    importance: float
    key_words: List[WordIG] = field(default_factory=list)


@dataclass
class IGAnalysisResult:
    """Full IG analysis result."""
    predicted_label: int
    predicted_label_name: str
    confidence: float
    sentence_results: List[SentenceIG] = field(default_factory=list)
    top_words: List[WordIG] = field(default_factory=list)


# ============================================================
# Stopword list
# (Bird, Klein, & Loper, 2009. Natural Language Processing with Python. O'Reilly)
# ============================================================

NLTK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",
}

SPECIAL_TOKENS = {'<unk>', '<sep>', '<pad>', '.', ',', '!', '?', '"', "'",
                  '-', ':', ';', '(', ')', '[', ']'}

PRE_DEFINED_STOPWORDS = {
    # Unicode punctuation (kept as-is since they are symbols)
    '—',   # em-dash (U+2014)
    '–',   # en-dash (U+2013)
    '\x97', # em-dash variant (U+0097)
    '"',  # left double quotation mark (U+201C)
    '"',  # right double quotation mark (U+201D)
    '--',  # double hyphen
    '|',   # pipe character
    '…',   # horizontal ellipsis (U+2026)
    # All entries below must be in lowercase.
    # They are compared against token.lower() during filtering.
    # Media credits / photo sources
    'getty', 'images',
    # Reporting boilerplate (journalist attribution)
    'associated', 'newsweek', 'reported', 'reporting', 'reached', 'said', 'says', 'according', 'publisher',
    # Transition words not in NLTK
    'though', 'thus', 'likewise', 'however', 'also', 'moreover', 'although',
    # In-article navigation elements
    'related', 'read', 'watch', 'breaking', 'updated',
    # Image UI
    'caption', 'hide',
    # Paywall UI
    'subscriber', 'login',
    # Symbols
    '©', '*', '&',
    # Unicode artifacts
    '​',  # zero-width space
    '﻿',  # BOM (byte order mark)
    # Fox News / news app UI
    'click', 'newsletter', 'app',
}

DEFAULT_IGNORE_TOKENS = NLTK_STOPWORDS | SPECIAL_TOKENS | PRE_DEFINED_STOPWORDS


# ============================================================
# IG analyzer
# ============================================================

class IntegratedGradientsAnalyzer:
    """Integrated Gradients analyzer for KHAN (KnowledgeEncoding target)."""

    def __init__(self, model, device: str = 'cpu', steps: int = 50):
        """
        Args:
            model: KHAN model (must have a knowledge_encoder attribute)
            device: 'cpu' or 'cuda'
            steps: Number of IG interpolation steps (higher = more precise, default 50)
        """
        self.model = model
        self.device = device
        self.steps = steps

    def analyze(
        self,
        sentences_tensor: torch.Tensor,
        titles_tensor: torch.Tensor,
        tokens: List[List[List[str]]],
        sentences: List[str],
        label_names: Dict[int, str],
        top_k_words: int = 10,
        top_k_per_sentence: int = 5,
    ) -> IGAnalysisResult:
        """
        Run Integrated Gradients XAI analysis.

        Args:
            sentences_tensor: [batch, num_sent, seq_len]
            titles_tensor: [batch, title_len]
            tokens: [batch][sent_idx][word_idx]
            sentences: List of original sentence strings
            label_names: {0: 'Left', 1: 'Lean Left', ...}
            top_k_words: Total number of top words to return
            top_k_per_sentence: Top words per sentence
        """
        self.model.eval()

        with torch.no_grad():
            output = self.model(sentences_tensor, titles_tensor, return_attention=False)

        probs = F.softmax(output, dim=-1)
        predicted_label = output.argmax(dim=-1).item()
        confidence = probs[0, predicted_label].item()

        ig_word_scores = self._compute_ig(sentences_tensor, titles_tensor, predicted_label)

        sentence_results: List[SentenceIG] = []
        all_words: List[WordIG] = []

        for sent_idx, sentence_text in enumerate(sentences):
            if not sentence_text.strip():
                continue

            sent_scores = ig_word_scores[sent_idx] if sent_idx < len(ig_word_scores) else np.array([])
            sent_importance = float(sent_scores.mean()) if len(sent_scores) > 0 else 0.0
            sent_tokens = self._get_sentence_tokens(tokens, sent_idx)

            word_results: List[WordIG] = []
            for word_idx, token in enumerate(sent_tokens):
                if token.lower() in DEFAULT_IGNORE_TOKENS or token == '<pad>':
                    continue
                if token.startswith('[') and token.endswith(']'):  # widget tags like [dcquiz]
                    continue
                if '@' in token:  # email addresses
                    continue
                score = float(sent_scores[word_idx]) if word_idx < len(sent_scores) else 0.0
                w = WordIG(word=token, score=score,
                           sentence_idx=sent_idx, position=word_idx)
                word_results.append(w)
                all_words.append(w)

            word_results.sort(key=lambda x: -x.score)
            sentence_results.append(SentenceIG(
                sentence_idx=sent_idx,
                text=sentence_text,
                importance=sent_importance,
                key_words=word_results[:top_k_per_sentence],
            ))

        all_words.sort(key=lambda x: -x.score)

        return IGAnalysisResult(
            predicted_label=predicted_label,
            predicted_label_name=label_names.get(predicted_label, str(predicted_label)),
            confidence=confidence,
            sentence_results=sentence_results,
            top_words=all_words[:top_k_words],
        )

    def _compute_ig(
        self,
        sentences_tensor: torch.Tensor,
        titles_tensor: torch.Tensor,
        target_class: int,
    ) -> List[np.ndarray]:
        """
        Manual IG computation using forward hooks, targeting KnowledgeEncoding output.

        Returns:
            List of [seq_len] numpy arrays per sentence (min-max normalized).
        """
        model = self.model
        ke_layer = model.knowledge_encoder

        # --- (1) Capture knowledge_encoder output for actual and baseline inputs ---
        ke_outputs: Dict[str, torch.Tensor] = {}

        def make_capture_hook(key):
            def hook(module, inp, output):
                ke_outputs[key] = output.detach().clone()
            return hook

        h = ke_layer.register_forward_hook(make_capture_hook('actual'))
        with torch.no_grad():
            model(sentences_tensor, titles_tensor)
        h.remove()

        baselines = torch.zeros_like(sentences_tensor)
        h = ke_layer.register_forward_hook(make_capture_hook('baseline'))
        with torch.no_grad():
            model(baselines, titles_tensor)
        h.remove()

        actual_ke = ke_outputs['actual']
        baseline_ke = ke_outputs['baseline']
        diff = actual_ke - baseline_ke

        # --- (2) Disable gradient tracking on model parameters (only interpolated needs grad) ---
        param_grad_states = {}
        for name, p in model.named_parameters():
            param_grad_states[name] = p.requires_grad
            p.requires_grad_(False)

        # --- (3) Accumulate gradients along the interpolation path (trapezoidal rule) ---
        accumulated_grads = torch.zeros_like(actual_ke)

        try:
            for step in range(self.steps + 1):
                alpha = step / self.steps
                interpolated = (baseline_ke + alpha * diff).detach().requires_grad_(True)

                def replace_hook(module, inp, output, interp=interpolated):
                    return interp

                h = ke_layer.register_forward_hook(replace_hook)
                output = model(sentences_tensor, titles_tensor)
                score = output[0, target_class]

                if interpolated.grad is not None:
                    interpolated.grad.zero_()
                score.backward()

                grad = interpolated.grad.detach().clone()
                if step == 0 or step == self.steps:
                    grad = grad * 0.5  # trapezoidal weight

                accumulated_grads += grad
                h.remove()
        finally:
            for name, p in model.named_parameters():
                p.requires_grad_(param_grad_states[name])
            model.eval()

        # --- (4) IG = diff * avg_grads → L2 norm → normalization ---
        avg_grads = accumulated_grads / self.steps
        ig_attributions = diff * avg_grads
        ig_scores = ig_attributions.cpu().norm(dim=-1).numpy()  # [num_sent, seq_len]

        gmin, gmax = ig_scores.min(), ig_scores.max()
        if gmax > gmin:
            ig_scores = (ig_scores - gmin) / (gmax - gmin)
        else:
            ig_scores = np.zeros_like(ig_scores)

        return [ig_scores[i] for i in range(ig_scores.shape[0])]

    def _get_sentence_tokens(self, tokens, sentence_idx):
        try:
            return tokens[0][sentence_idx]
        except (IndexError, TypeError):
            return []
