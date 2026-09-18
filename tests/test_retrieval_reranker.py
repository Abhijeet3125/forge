# tests/test_retrieval_reranker.py

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.retrieval.chunking import CodeChunk
from core.retrieval.reranker import Reranker, format_chunk_for_reranker
from core.tools.search_codebase import retrieve_context


class MockScoringReranker(Reranker):
    """Deterministic reranker verifying rank promotion and scoring."""

    def __init__(self):
        super().__init__(model_name="mock-reranker")

    def rerank_with_scores(self, query: str, chunks: list[CodeChunk], top_k: int = 3):
        scored = []
        for c in chunks:
            s = 0.0
            if c.name.lower() in query.lower():
                s += 0.6
            if Path(c.file_path).name.lower() in query.lower():
                s += 0.3
            scored.append((s, c))
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        return ranked[:top_k]


class TestRerankerPipeline(unittest.TestCase):
    def test_format_chunk_for_reranker(self):
        chunk = CodeChunk(
            chunk_id="c1",
            file_path="src/cart.py",
            name="calculate_total",
            kind="function",
            start_line=1,
            end_line=5,
            content="def calculate_total(): pass",
            signature="def calculate_total():",
            docstring="Calculates total.",
        )
        formatted = format_chunk_for_reranker(chunk)
        self.assertIn("File: src/cart.py", formatted)
        self.assertIn("Symbol: calculate_total (function)", formatted)
        self.assertIn("Signature: def calculate_total():", formatted)
        self.assertIn("Docstring: Calculates total.", formatted)
        self.assertIn("def calculate_total(): pass", formatted)

    def test_reranker_promotes_target_over_distractor(self):
        query = "Fix calculate_total in cart.py that fails to apply discount"

        distractor_chunk = CodeChunk(
            chunk_id="distractor_analytics",
            file_path="analytics/reports.py",
            name="calculate_discount_totals",
            kind="function",
            start_line=1,
            end_line=10,
            content="def calculate_discount_totals(): pass",
        )

        target_chunk = CodeChunk(
            chunk_id="target_cart",
            file_path="cart.py",
            name="calculate_total",
            kind="function",
            start_line=1,
            end_line=10,
            content="def calculate_total(prices, discount): pass",
        )

        # Initial candidates order: distractor is at Rank 1, target is at Rank 2
        candidates = [distractor_chunk, target_chunk]

        reranker = MockScoringReranker()
        reranked = reranker.rerank(query=query, chunks=candidates, top_k=2)

        # Confirm target is promoted to Rank 1
        self.assertEqual(reranked[0].chunk_id, "target_cart")
        self.assertEqual(reranked[1].chunk_id, "distractor_analytics")


if __name__ == "__main__":
    unittest.main()
