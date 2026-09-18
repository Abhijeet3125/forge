# core/retrieval/reranker.py

from typing import Any, cast
from core.retrieval.chunking import CodeChunk

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def format_chunk_for_reranker(chunk: CodeChunk) -> str:
    """
    Enriches candidate code text with metadata (file path, symbol name, signature)
    so the CrossEncoder can evaluate relevance with full structural context.
    """
    header_parts = [f"File: {chunk.file_path}"]
    if chunk.name:
        header_parts.append(f"Symbol: {chunk.name} ({chunk.kind})")
    if chunk.signature:
        header_parts.append(f"Signature: {chunk.signature}")
    if chunk.docstring:
        header_parts.append(f"Docstring: {chunk.docstring}")

    header = " | ".join(header_parts)
    return f"{header}\n\n{chunk.content}"


class Reranker:
    """
    Wrapper around CrossEncoder (bge-reranker-base) for re-scoring
    candidate CodeChunks retrieved from ChromaDB down to top-3.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank_with_scores(
        self,
        query: str,
        chunks: list[CodeChunk],
        top_k: int = 3,
    ) -> list[tuple[float, CodeChunk]]:
        """
        Scores (query, enriched_chunk) pairs and returns list of (score, chunk)
        sorted by relevance score descending.
        """
        if not chunks:
            return []

        try:
            model = self._load_model()
            pairs = [(query, format_chunk_for_reranker(chunk)) for chunk in chunks]
            scores = model.predict(cast(Any, pairs))

            # Zip and sort descending
            ranked = sorted(zip(scores, chunks), key=lambda x: float(x[0]), reverse=True)
            return [(float(score), chunk) for score, chunk in ranked[:top_k]]
        except Exception:
            # Fallback: assign mock decreasing scores matching original vector rank
            return [(1.0 - (i * 0.05), chunk) for i, chunk in enumerate(chunks[:top_k])]

    def rerank(self, query: str, chunks: list[CodeChunk], top_k: int = 3) -> list[CodeChunk]:
        """
        Scores (query, chunk) pairs and returns top_k CodeChunks.
        """
        scored = self.rerank_with_scores(query=query, chunks=chunks, top_k=top_k)
        return [chunk for _, chunk in scored]


_default_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = Reranker()
    return _default_reranker
