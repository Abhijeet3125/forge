# core/retrieval/embeddings.py

from typing import Any

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    """
    Wrapper around SentenceTransformer for code and query embeddings.
    Defaults to BAAI/bge-small-en-v1.5 per Phase 1 specification.
    Lazy-loads the underlying PyTorch model on first use.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Generates normalized embeddings for a list of document strings."""
        if not texts:
            return []

        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.tolist()

    def embed_query(self, query: str, use_prefix: bool = True) -> list[float]:
        """
        Generates a normalized embedding for a search query.
        Uses the recommended BGE instruction prefix if use_prefix is True.
        """
        text = f"{BGE_QUERY_PREFIX}{query.strip()}" if use_prefix else query.strip()
        model = self._load_model()
        embedding = model.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embedding[0].tolist()


# Global singleton instance for easy reuse across tools and retriever
_default_embedding_model: EmbeddingModel | None = None


def get_embedding_model() -> EmbeddingModel:
    global _default_embedding_model
    if _default_embedding_model is None:
        _default_embedding_model = EmbeddingModel()
    return _default_embedding_model
