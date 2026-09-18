#!/usr/bin/env python3
# scripts/index_repo.py

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.retrieval.indexer import RepoIndexer
from core.retrieval.vectorstore import VectorStore


def main():
    parser = argparse.ArgumentParser(
        description="One-time repository indexer for Forge coding agent retrieval.",
    )
    parser.add_argument(
        "repo_path",
        type=str,
        help="Path to the repository to index (e.g. . or /path/to/repo)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="chroma_db",
        help="Path to ChromaDB persistent storage directory (default: chroma_db)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="codebase_chunks",
        help="Name of the ChromaDB collection (default: codebase_chunks)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing collection before indexing",
    )

    args = parser.parse_args()

    target_path = Path(args.repo_path).resolve()
    if not target_path.exists() or not target_path.is_dir():
        print(f"Error: Target path '{args.repo_path}' does not exist or is not a directory.")
        sys.exit(1)

    print(f"Indexing repository: {target_path}")
    print(f"Persistent storage:  {args.db_path} (collection: {args.collection})")

    vs = VectorStore(db_path=args.db_path, collection_name=args.collection)
    if args.reset:
        print("Resetting existing collection...")
        vs.reset()

    def on_progress(current: int, total: int):
        if current % 10 == 0 or current == total:
            print(f"  Scanned {current}/{total} source files...", end="\r", flush=True)

    indexer = RepoIndexer(repo_path=str(target_path), vector_store=vs)
    stats = indexer.index(progress_callback=on_progress)

    print()
    print("=" * 45)
    print("Indexing Complete!")
    print(f"  Files scanned:    {stats.files_scanned}")
    print(f"  Chunks extracted: {stats.chunks_extracted}")
    print(f"  Chunks indexed:   {stats.chunks_indexed}")
    print(f"  Total chunks in DB: {vs.count()}")
    print(f"  Elapsed time:     {stats.elapsed_seconds:.2f}s")
    print("=" * 45)


if __name__ == "__main__":
    main()
