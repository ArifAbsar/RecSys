"""
semantic_classifier.py
──────────────────────
Intelligent role classification using semantic embeddings.
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticRoleClassifier:
    """
    Intelligent role classification using semantic embeddings.
    Compares column meanings against conceptual anchors or custom phrases.
    """
    def __init__(
        self, 
        anchors: Dict[str, List[str]], 
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)
        self._anchor_embeddings = {}
        self._anchors = anchors
        self._fit_anchors()

    def _fit_anchors(self):
        """Pre-compute embeddings for all conceptual anchors."""
        if not self._anchors:
            return

        for role, phrases in self._anchors.items():
            embeddings = self.model.encode(phrases, normalize_embeddings=True)
            self._anchor_embeddings[role] = np.mean(embeddings, axis=0)

    def classify(self, text: str, custom_anchors: Optional[List[str]] = None) -> tuple[str, float]:
        """
        Compare the input text against anchors and return the best match.
        If custom_anchors is provided, it compares against those phrases instead.
        """
        input_embedding = self.model.encode([text], normalize_embeddings=True)

        if custom_anchors:
            # Compare against specific phrases provided in the rule
            anchor_embeddings = self.model.encode(custom_anchors, normalize_embeddings=True)
            scores = cosine_similarity(input_embedding, anchor_embeddings)[0]
            best_idx = np.argmax(scores)
            return custom_anchors[best_idx], float(scores[best_idx])

        if not self._anchor_embeddings:
            return "unknown", 0.0
        
        roles = list(self._anchor_embeddings.keys())
        anchors = np.array([self._anchor_embeddings[r] for r in roles])
        
        scores = cosine_similarity(input_embedding, anchors)[0]
        
        best_idx = np.argmax(scores)
        role = roles[best_idx]
        confidence = float(scores[best_idx])
        
        return role, confidence
