import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from Mapping_Engine.text_builder.text_builder import build_source_text, build_target_text

class Embedding:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.targets = []
        self.target_embeddings = None

    def fit_targets(self, target: list[dict]):
        self.targets = target
        target_texts = [build_target_text(text) for text in target]
        self.target_embeddings = self.model.encode(
            target_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def match_column(self, source_column: dict) -> list[dict]:
        if self.target_embeddings is None:
            raise ValueError("Run fit_targets() first.")

        source_text = build_source_text(source_column)
        source_embedding = self.model.encode([source_text], normalize_embeddings=True)
        scores = cosine_similarity(source_embedding, self.target_embeddings)[0]

        sorted_indices = np.argsort(scores)[::-1]

        return [
            {
                **self.targets[i],
                "score": float(scores[i]),  
            }
            for i in sorted_indices
        ]
