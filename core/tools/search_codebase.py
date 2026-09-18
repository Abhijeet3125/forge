# core/tools/search_codebase.py

from langchain_core.tools import tool

from core.retrieval.chunking import CodeChunk
from core.retrieval.reranker import Reranker, get_reranker
from core.retrieval.vectorstore import VectorStore


def retrieve_context(
    query: str,
    vector_store: VectorStore | None = None,
    reranker: Reranker | None = None,
    top_k_initial: int = 20,
    top_k_final: int = 3,
) -> list[CodeChunk]:
    """
    Executes the 2-stage retrieval pipeline:
    1. Query ChromaDB for top-20 candidate chunks.
    2. Rerank down to top-3 using bge-reranker.
    """
    vs = vector_store or VectorStore()
    rr = reranker or get_reranker()

    candidates = vs.query(query_text=query, top_k=top_k_initial)
    if not candidates:
        return []

    return rr.rerank(query=query, chunks=candidates, top_k=top_k_final)


def format_chunks_for_llm(chunks: list[CodeChunk]) -> str:
    """Formats retrieved CodeChunks into a concise, compacted string for the agent."""
    if not chunks:
        return "No relevant code found in the codebase index."

    sections = []
    for i, chunk in enumerate(chunks, 1):
        header = f"### [{i}] {chunk.file_path} (Lines {chunk.start_line}-{chunk.end_line}) | {chunk.kind}: {chunk.name}"
        if chunk.signature:
            header += f"\nSignature: `{chunk.signature}`"

        code_block = f"```python\n{chunk.content.strip()}\n```"
        sections.append(f"{header}\n{code_block}")

    return "\n\n".join(sections)


def make_search_codebase_tool(db_path: str = "chroma_db"):
    """
    Factory that binds the vector database path to the search_codebase tool.
    Used by both the Planner and Coder agents.
    """
    vs = VectorStore(db_path=db_path)
    rr = get_reranker()

    @tool
    def search_codebase(query: str) -> str:
        """
        Semantically search the repository codebase for relevant functions, classes, and code chunks.
        Returns the top most relevant snippets with file paths and line numbers.

        Args:
            query: search query describing the functionality or bug, e.g. 'calculate total cart discount'
        """
        chunks = retrieve_context(query, vector_store=vs, reranker=rr)
        return format_chunks_for_llm(chunks)

    return search_codebase
