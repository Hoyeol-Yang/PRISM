"""
spaCy-based keyword expander.

Uses dependency parsing to expand single-word keywords
into meaningful phrase units.
"""

import spacy
from typing import List, Dict, Tuple, Optional
from .models import ExpandedKeyword


class KeywordExpander:
    """
    spaCy-based keyword expander.

    Expansion rules:
    - Adjective (ADJ) → paired with the noun it modifies (e.g., radical → radical policy)
    - Noun (NOUN) → paired with its adjective modifiers (e.g., tax → progressive tax)
    - Verb (VERB) → paired with its object (e.g., attack → attack opponents)
    """

    def __init__(self, model: str = "en_core_web_lg"):
        """
        Args:
            model: spaCy model name (default: en_core_web_lg)
        """
        print(f"  Loading spaCy model: {model}...")
        self.nlp = spacy.load(model)
        print(f"  spaCy loaded successfully")

    def expand_keywords(
        self,
        text: str,
        keywords: List[Dict],
        top_k: int = 10
    ) -> List[ExpandedKeyword]:
        """
        Expand keywords to phrase units.

        Args:
            text: Original article body
            keywords: Keywords extracted by XAI
                      [{"word": "radical", "gradient": 0.89, "attention": 0.75, "combined": 0.67}, ...]
            top_k: Number of top keywords to expand

        Returns:
            List[ExpandedKeyword]: List of expanded keywords
        """
        # Replace <SEP> tokens with periods so spaCy parses sentences correctly
        clean_text = text.replace("<SEP>", ".")
        doc = self.nlp(clean_text)

        # Keyword set for fast lookup
        keyword_set = {kw["word"].lower() for kw in keywords[:top_k]}
        keyword_scores = {kw["word"].lower(): kw for kw in keywords[:top_k]}

        expanded_results = []
        processed_keywords = set()  # deduplication

        for token in doc:
            token_lower = token.text.lower()

            # Skip if already processed or not a keyword
            if token_lower in processed_keywords:
                continue
            if token_lower not in keyword_set:
                continue

            # Perform expansion
            expanded, exp_type = self._expand_token(token)

            # Clean up unnecessary characters from expanded phrase
            expanded = self._clean_expanded(expanded)

            # Find the sentence containing the keyword
            context = self._find_context_sentence(token)

            # Retrieve scores
            scores = keyword_scores.get(token_lower, {})

            expanded_kw = ExpandedKeyword(
                original=token.text,
                expanded=expanded,
                expansion_type=exp_type,
                original_pos=token.pos_,
                context_sentence=context,
                scores={
                    "gradient": scores.get("gradient", 0.0),
                    "attention": scores.get("attention", 0.0),
                    "combined": scores.get("combined", 0.0)
                }
            )

            expanded_results.append(expanded_kw)
            processed_keywords.add(token_lower)

        # Keywords not matched by spaCy are kept as expanded = original
        for kw in keywords[:top_k]:
            if kw["word"].lower() not in processed_keywords:
                scores = keyword_scores.get(kw["word"].lower(), kw)
                expanded_results.append(ExpandedKeyword(
                    original=kw["word"],
                    expanded=kw["word"],
                    expansion_type="ORIGINAL",
                    original_pos="",
                    context_sentence="",
                    scores={
                        "gradient": scores.get("gradient", 0.0),
                        "attention": scores.get("attention", 0.0),
                        "combined": scores.get("combined", 0.0),
                    }
                ))

        # Sort by combined score
        expanded_results.sort(key=lambda x: x.scores.get("combined", 0), reverse=True)

        return expanded_results

    def _clean_expanded(self, expanded: str) -> str:
        """Remove unnecessary characters from the expanded phrase."""
        import re
        expanded = re.sub(r'\.+', '', expanded)
        expanded = re.sub(r'\s+', ' ', expanded)
        return expanded.strip()

    def _expand_token(self, token) -> Tuple[str, str]:
        """
        Convert a token into an expanded phrase.

        Returns:
            (expanded phrase, expansion type)
        """
        pos = token.pos_

        # Adjective: find the noun it modifies (head)
        if pos == "ADJ":
            return self._expand_adjective(token)

        # Noun: find its adjective modifiers (children)
        elif pos == "NOUN":
            return self._expand_noun(token)

        # Verb: find its object
        elif pos == "VERB":
            return self._expand_verb(token)

        # Other: no expansion
        else:
            return token.text, f"{pos} → {pos}"

    def _expand_adjective(self, token) -> Tuple[str, str]:
        """Adjective expansion: include the noun it modifies."""
        head = token.head

        # amod relation where head is a noun
        if token.dep_ == "amod" and head.pos_ == "NOUN":
            # Collect other adjective modifiers of the same noun
            modifiers = [token]
            for child in head.children:
                if child.pos_ == "ADJ" and child != token:
                    modifiers.append(child)

            # Sort by position
            modifiers.sort(key=lambda x: x.i)

            phrase_parts = [m.text for m in modifiers] + [head.text]
            phrase = " ".join(phrase_parts)

            if len(modifiers) > 1:
                exp_type = f"ADJ → ADJ+ADJ+NOUN"
            else:
                exp_type = f"ADJ → ADJ+NOUN"

            return phrase, exp_type

        return token.text, "ADJ → ADJ"

    def _expand_noun(self, token) -> Tuple[str, str]:
        """Noun expansion: include adjective modifiers."""
        modifiers = []

        # Collect adjective modifiers
        for child in token.children:
            if child.pos_ == "ADJ" and child.dep_ == "amod":
                modifiers.append(child)

        if modifiers:
            modifiers.sort(key=lambda x: x.i)
            phrase_parts = [m.text for m in modifiers] + [token.text]
            phrase = " ".join(phrase_parts)

            if len(modifiers) > 1:
                exp_type = f"NOUN → ADJ+ADJ+NOUN"
            else:
                exp_type = f"NOUN → ADJ+NOUN"

            return phrase, exp_type

        # Check for compound noun (compound relation)
        compounds = []
        for child in token.children:
            if child.dep_ == "compound":
                compounds.append(child)

        if compounds:
            compounds.sort(key=lambda x: x.i)
            phrase_parts = [c.text for c in compounds] + [token.text]
            phrase = " ".join(phrase_parts)
            return phrase, f"NOUN → COMPOUND+NOUN"

        return token.text, "NOUN → NOUN"

    def _expand_verb(self, token) -> Tuple[str, str]:
        """Verb expansion: include the object."""
        # Direct object
        for child in token.children:
            if child.dep_ == "dobj":
                phrase = f"{token.text} {child.text}"
                return phrase, "VERB → VERB+OBJ"

        # Prepositional phrase
        for child in token.children:
            if child.dep_ == "prep":
                for grandchild in child.children:
                    if grandchild.dep_ == "pobj":
                        phrase = f"{token.text} {child.text} {grandchild.text}"
                        return phrase, "VERB → VERB+PREP+NOUN"

        return token.text, "VERB → VERB"

    def _find_context_sentence(self, token) -> str:
        """Return the sentence containing the token."""
        sent = token.sent
        if not sent:
            return ""

        return sent.text.strip()
