"""
Keyword Expansion Module

Expands IG keywords to phrase units using spaCy dependency parsing
and saves results in JSON format.

Usage:
    from keyword_expander import KeywordExpander, XAIJsonExporter

    expander = KeywordExpander()
    exporter = XAIJsonExporter()
"""

from .models import ExpandedKeyword
from .keyword_expander import KeywordExpander
from .json_exporter import XAIJsonExporter

__all__ = [
    'ExpandedKeyword',
    'KeywordExpander', 
    'XAIJsonExporter'
]
