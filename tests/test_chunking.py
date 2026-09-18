# tests/test_chunking.py

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.retrieval.chunking import (
    CodeChunk,
    chunk_file_content,
    _chunk_python_ast,
    LARGE_FUNCTION_LINE_THRESHOLD,
)


class TestChunking(unittest.TestCase):
    def test_chunk_functions_and_classes(self):
        code = (
            "class ShoppingCart:\n"
            '    """Manages items in cart."""\n'
            "    def __init__(self):\n"
            "        self.items = []\n"
            "\n"
            "    def add_item(self, item: str) -> None:\n"
            '        """Add an item."""\n'
            "        self.items.append(item)\n"
            "\n"
            "def calculate_tax(subtotal: float) -> float:\n"
            '    """Calculate 10% tax."""\n'
            "    return subtotal * 0.1\n"
        )

        chunks = chunk_file_content(code, "cart.py")
        names = [c.name for c in chunks]

        self.assertIn("ShoppingCart", names)
        self.assertIn("ShoppingCart.add_item", names)
        self.assertIn("calculate_tax", names)

        # Check metadata
        tax_chunk = next(c for c in chunks if c.name == "calculate_tax")
        self.assertEqual(tax_chunk.kind, "function")
        self.assertIsNotNone(tax_chunk.signature)
        assert tax_chunk.signature is not None
        self.assertTrue(tax_chunk.signature.startswith("def calculate_tax"))

    def test_large_function_summary_chunk(self):
        # Generate a function with more lines than threshold
        body = "\n".join([f"    x_{i} = {i}" for i in range(LARGE_FUNCTION_LINE_THRESHOLD + 10)])
        code = (
            "def very_long_function(a, b):\n"
            '    """A lengthy processing function."""\n'
            f"{body}\n"
            "    return a + b\n"
        )

        chunks = chunk_file_content(code, "long.py")
        kinds = [c.kind for c in chunks]

        self.assertIn("function", kinds)
        self.assertIn("summary", kinds)

        summary_chunk = next(c for c in chunks if c.kind == "summary")
        self.assertIn("def very_long_function", summary_chunk.content)
        self.assertIn("A lengthy processing function.", summary_chunk.content)

    def test_syntax_error_resilience(self):
        broken_code = "def unclosed_func(:\n    broken syntax !!!"
        chunks = chunk_file_content(broken_code, "broken.py")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].kind, "module")
        self.assertEqual(chunks[0].name, "module")

    def test_chunk_metadata_serialization(self):
        chunk = CodeChunk(
            chunk_id="test_id_123",
            file_path="src/util.py",
            name="format_date",
            kind="function",
            start_line=10,
            end_line=25,
            content="def format_date(d): return str(d)",
            signature="def format_date(d):",
            docstring="Formats a date.",
        )

        meta = chunk.to_metadata()
        restored = CodeChunk.from_metadata(chunk.content, meta)

        self.assertEqual(restored.chunk_id, chunk.chunk_id)
        self.assertEqual(restored.file_path, chunk.file_path)
        self.assertEqual(restored.name, chunk.name)
        self.assertEqual(restored.kind, chunk.kind)
        self.assertEqual(restored.start_line, chunk.start_line)
        self.assertEqual(restored.end_line, chunk.end_line)
        self.assertEqual(restored.signature, chunk.signature)
        self.assertEqual(restored.docstring, chunk.docstring)
        self.assertEqual(restored.content, chunk.content)


if __name__ == "__main__":
    unittest.main()
