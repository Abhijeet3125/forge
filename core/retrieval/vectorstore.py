# core/retrieval/vectorstore.py

from pathlib import Path
from typing import Any, cast

from core.retrieval.chunking import CodeChunk
from core.retrieval.embeddings import EmbeddingModel, get_embedding_model

DEFAULT_CHROMA_PATH = "chroma_db"
DEFAULT_COLLECTION_NAME = "codebase_chunks"


class VectorStore:
    """
    ChromaDB wrapper for indexing and semantic retrieval of CodeChunks.
    Persists vectors and metadata locally in ChromaDB.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_CHROMA_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.db_path = str(Path(db_path).resolve())
        self.collection_name = collection_name
        self.embedding_model = embedding_model or get_embedding_model()
        self._client: Any = None
        self._collection: Any = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            # Ensure directory exists
            Path(self.db_path).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.db_path)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[CodeChunk], batch_size: int = 64) -> int:
        """
        Computes embeddings and upserts CodeChunks into ChromaDB.
        Deduplicates chunks by chunk_id within the batch.
        """
        if not chunks:
            return 0

        collection = self._get_collection()

        # Deduplicate chunks by ID
        unique_chunks: dict[str, CodeChunk] = {}
        for c in chunks:
            unique_chunks[c.chunk_id] = c
        chunk_list = list(unique_chunks.values())

        total_added = 0
        for i in range(0, len(chunk_list), batch_size):
            batch = chunk_list[i : i + batch_size]
            contents = [c.content for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [c.to_metadata() for c in batch]

            embeddings = self.embedding_model.embed_documents(contents)

            collection.upsert(
                ids=ids,
                documents=contents,
                embeddings=cast(Any, embeddings),
                metadatas=cast(Any, metadatas),
            )
            total_added += len(batch)

        return total_added

    def query(self, query_text: str, top_k: int = 20) -> list[CodeChunk]:
        """
        Embeds the query text and retrieves the top_k nearest CodeChunks
        from ChromaDB using cosine similarity.
        """
        collection = self._get_collection()
        total_items = collection.count()
        if total_items == 0:
            return []

        actual_k = min(top_k, total_items)
        query_embedding = self.embedding_model.embed_query(query_text)

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        chunks: list[CodeChunk] = []
        documents = results.get("documents")
        metadatas = results.get("metadatas")
        docs = documents[0] if documents else []
        metas = metadatas[0] if metadatas else []

        for doc, meta in zip(docs, metas):
            if doc and meta:
                chunks.append(CodeChunk.from_metadata(content=doc, meta=meta))

        return chunks

    def count(self) -> int:
        """Returns the total number of chunks in the collection."""
        return self._get_collection().count()

    def reset(self) -> None:
        """Clears all records in the active collection."""
        if self._client is not None:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = None
