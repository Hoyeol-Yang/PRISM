"""
XAI result JSON exporter.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from .models import ExpandedKeyword


class XAIJsonExporter:
    """
    Saves XAI analysis results in JSON format.

    Produces JSON output only — no HTML.
    """

    def __init__(self, output_dir: str = None):
        """
        Args:
            output_dir: Output directory path (default: outputs/ig/allsides-l/)
        """
        if output_dir is None:
            # keyword_expander/ → IG_khan/ → Unified/
            unified_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(unified_root, "outputs", "IG_khan")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export(
        self,
        title: str,
        text: str,
        prediction_label: str,
        confidence: float,
        expanded_keywords: List[ExpandedKeyword],
        dataset: str = "unknown",
        article_id: Optional[int] = None,
        probabilities: Optional[Dict[str, float]] = None
    ) -> dict:
        """
        Convert analysis results to a dictionary.

        Args:
            title: Article title
            text: Article body
            prediction_label: Predicted political bias label
            confidence: Prediction confidence score
            expanded_keywords: List of expanded keywords
            dataset: Dataset name
            article_id: Article ID (optional)
            probabilities: Per-label probabilities (e.g., {"Left": 0.87, "Center": 0.09, "Right": 0.04})

        Returns:
            JSON-serializable dictionary
        """
        prediction = {
            "label": prediction_label,
        }
        if probabilities:
            prediction["probabilities"] = probabilities

        return {
            "article": {
                "id": article_id,
                "title": title,
                "text": text
            },
            "prediction": prediction,
            "keywords": [kw.to_dict() for kw in expanded_keywords],
            "metadata": {
                "model": "KHAN",
                "dataset": dataset,
                "analysis_timestamp": datetime.now().isoformat(),
                "keyword_count": len(expanded_keywords)
            }
        }

    def save(self, data: dict, filename: str) -> str:
        """
        Save to a JSON file.

        Args:
            data: Dictionary returned by export()
            filename: Filename without extension

        Returns:
            Path to the saved file
        """
        if not filename.endswith('.json'):
            filename = f"{filename}.json"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"  JSON saved: {filepath}")
        return filepath

    def export_and_save(
        self,
        title: str,
        text: str,
        prediction_label: str,
        confidence: float,
        expanded_keywords: List[ExpandedKeyword],
        filename: str,
        dataset: str = "unknown",
        article_id: Optional[int] = None,
        probabilities: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Run export and save in a single call.

        Returns:
            Path to the saved file
        """
        data = self.export(
            title=title,
            text=text,
            prediction_label=prediction_label,
            confidence=confidence,
            expanded_keywords=expanded_keywords,
            dataset=dataset,
            article_id=article_id,
            probabilities=probabilities
        )
        return self.save(data, filename)
