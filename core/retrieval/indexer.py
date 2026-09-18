# core/retrieval/indexer.py

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.retrieval.chunking import CodeChunk, chunk_file
from core.retrieval.vectorstore import VectorStore

DEFAULT_IGNORED_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "chroma_db",
    ".idea",
    ".vscode",
}

DEFAULT_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs"}


@dataclass
class IndexStats:
    files_scanned: int
    chunks_extracted: int
    chunks_indexed: int
    elapsed_seconds: float = 0.0


class RepoIndexer:
    """
    Top-level repository indexing pipeline:
    Traverses the target repo, extracts AST chunks, and indexes them into ChromaDB.
    """

    def __init__(
        self,
        repo_path: str,
        vector_store: VectorStore | None = None,
        ignored_dirs: set[str] | None = None,
        extensions: set[str] | None = None,
    ):
        self.repo_root = Path(repo_path).resolve()
        self.vector_store = vector_store or VectorStore()
        self.ignored_dirs = ignored_dirs if ignored_dirs is not None else DEFAULT_IGNORED_DIRS
        self.extensions = extensions if extensions is not None else DEFAULT_EXTENSIONS

    def collect_source_files(self) -> list[Path]:
        """Collects all source files matching extensions while skipping ignored directories."""
        source_files: list[Path] = []

        if not self.repo_root.exists() or not self.repo_root.is_dir():
            return source_files

        for path in self.repo_root.rglob("*"):
            if path.is_file():
                # Check if any parent part is in ignored_dirs
                rel_parts = path.relative_to(self.repo_root).parts
                if any(part in self.ignored_dirs for part in rel_parts[:-1]):
                    continue

                if path.suffix.lower() in self.extensions:
                    source_files.append(path)

        return sorted(source_files)

    def index(self, progress_callback: Callable[[int, int], None] | None = None) -> IndexStats:
        """
        Indexes the repository into the vector store.
        Returns IndexStats with counts of scanned files and indexed chunks.
        """
        import time

        start_time = time.time()
        files = self.collect_source_files()
        all_chunks: list[CodeChunk] = []

        for i, file_path in enumerate(files):
            file_chunks = chunk_file(file_path, repo_root=self.repo_root)
            all_chunks.extend(file_chunks)
            if progress_callback:
                progress_callback(i + 1, len(files))

        chunks_added = self.vector_store.add_chunks(all_chunks)

        return IndexStats(
            files_scanned=len(files),
            chunks_extracted=len(all_chunks),
            chunks_indexed=chunks_added,
            elapsed_seconds=time.time() - start_time,
        )


def index_repository(repo_path: str, db_path: str = "chroma_db") -> IndexStats:
    """Convenience function to index a repository into ChromaDB."""
    vs = VectorStore(db_path=db_path)
    indexer = RepoIndexer(repo_path=repo_path, vector_store=vs)
    return indexer.index()
