"""
Data model definitions.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ExpandedKeyword:
    """Expanded keyword information."""

    original: str
    expanded: str
    expansion_type: str
    original_pos: str
    context_sentence: str = ""
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "original": self.original,
            "expanded": self.expanded,
            "expansion_type": self.expansion_type,
            "original_pos": self.original_pos,
            "context_sentence": self.context_sentence,
            "scores": self.scores
        }
