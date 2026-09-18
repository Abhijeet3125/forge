# tests/test_vectorstore.py

from pathlib import Path
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.retrieval.chunking import CodeChunk
from core.retrieval.embeddings import EmbeddingModel
from core.retrieval.vectorstore import VectorStore


class MockEmbeddingModel(EmbeddingModel):
    """Deterministic, lightweight embedding model for unit testing without downloading torch weights."""

    def __init__(self, dim: int = 64):
        super().__init__(model_name="mock-model")
        self.dim = dim

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self._hash_text(t) for t in texts]

    def embed_query(self, query: str, use_prefix: bool = True) -> list[float]:
        return self._hash_text(query)

    def _hash_text(self, text: str) -> list[float]:
        import hashlib
        import math

        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [(b / 255.0) - 0.5 for b in h[: self.dim]]
        # Normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = self.temp_dir.name
        self.mock_emb = MockEmbeddingModel()
        self.vs = VectorStore(
            db_path=self.db_path,
            collection_name="test_collection",
            embedding_model=self.mock_emb,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_add_and_count_chunks(self):
        chunks = [
            CodeChunk(
                chunk_id="chunk_1",
                file_path="src/math_util.py",
                name="add",
                kind="function",
                start_line=1,
                end_line=5,
                content="def add(a, b): return a + b",
                signature="def add(a, b):",
            ),
            CodeChunk(
                chunk_id="chunk_2",
                file_path="src/math_util.py",
                name="sub",
                kind="function",
                start_line=6,
                end_line=10,
                content="def sub(a, b): return a - b",
                signature="def sub(a, b):",
            ),
        ]

        added = self.vs.add_chunks(chunks)
        self.assertEqual(added, 2)
        self.assertEqual(self.vs.count(), 2)

    def test_query_chunks(self):
        chunk = CodeChunk(
            chunk_id="calc_total",
            file_path="src/cart.py",
            name="calculate_total",
            kind="function",
            start_line=15,
            end_line=30,
            content="def calculate_total(items): return sum(items)",
            signature="def calculate_total(items):",
            docstring="Computes sum of items in cart.",
        )
        self.vs.add_chunks([chunk])

        results = self.vs.query("calculate_total", top_k=5)
        self.assertGreaterEqual(len(results), 1)

        first = results[0]
        self.assertEqual(first.chunk_id, "calc_total")
        self.assertEqual(first.file_path, "src/cart.py")
        self.assertEqual(first.name, "calculate_total")
        self.assertEqual(first.docstring, "Computes sum of items in cart.")

    def test_deduplication_and_reset(self):
        chunk = CodeChunk(
            chunk_id="unique_1",
            file_path="app.py",
            name="run",
            kind="function",
            start_line=1,
            end_line=2,
            content="def run(): pass",
        )

        self.vs.add_chunks([chunk])
        self.assertEqual(self.vs.count(), 1)

        # Upserting same ID does not create duplicate
        self.vs.add_chunks([chunk])
        self.assertEqual(self.vs.count(), 1)

        self.vs.reset()
        self.assertEqual(self.vs.count(), 0)


if __name__ == "__main__":
    unittest.main()
