"""
search_policies: semantic search over the TeleNova policy documents.

Follows the shared tool contract (shared_schema.json -> tool_contract):
the function never raises, it always returns a success or an error dict.

Usage from the agent:
    from vector_db.search import search_policies

Self-test (from the repo root):  python -m vector_db.search
"""
import chromadb
from sentence_transformers import SentenceTransformer

from config import CHROMA_STORE_DIR
from vector_db.settings import COLLECTION_NAME, EMBEDDING_MODEL

TOOL_NAME = "search_policies"
_model = None
_collection = None


def _error(error_type: str, message: str, missing_fields=None) -> dict:
    return {
        "status": "error",
        "tool": TOOL_NAME,
        "data": None,
        "error": {
            "type": error_type,
            "message": message,
            "missing_fields": missing_fields or [],
        },
    }


def _load():
    """Load the embedding model and the collection once, then reuse them."""
    global _model, _collection
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    if _collection is None:
        if not CHROMA_STORE_DIR.exists():
            raise FileNotFoundError(
                "Vector store not found. Run python -m vector_db.build_vectordb first."
            )
        client = chromadb.PersistentClient(path=str(CHROMA_STORE_DIR))
        _collection = client.get_collection(COLLECTION_NAME)


def search_policies(query: str, top_k: int = 3) -> dict:
    """Return the policy passages most similar in meaning to `query`."""
    try:
        if not isinstance(query, str) or not query.strip():
            return _error("validation_error",
                          "query must be a non-empty string", ["query"])
        if not isinstance(top_k, int) or isinstance(top_k, bool) \
                or not 1 <= top_k <= 10:
            return _error("validation_error",
                          "top_k must be an integer between 1 and 10")

        try:
            _load()
        except FileNotFoundError as e:
            return _error("connection_error", str(e))

        embedding = _model.encode([query], normalize_embeddings=True).tolist()
        res = _collection.query(query_embeddings=embedding, n_results=top_k)

        documents = res["documents"][0]
        if not documents:
            return _error("not_found", "No relevant policy passages found")

        results = [
            {"text": text, "source_file": meta["source_file"],
             "chunk_id": meta["chunk_id"], "topic": meta["topic"],
             "distance": round(dist, 4)}
            for text, meta, dist in zip(
                documents, res["metadatas"][0], res["distances"][0]
            )
        ]
        return {
            "status": "ok",
            "tool": TOOL_NAME,
            "data": {"query": query, "results": results},
            "error": None,
        }
    except Exception as e:  # tools never raise
        return _error("internal_error", str(e))


if __name__ == "__main__":
    import json

    tests = [
        "How much is the early termination fee for a One year contract?",
        "What offers exist for long-term customers?",
        "Which plans can I get with a Month-to-month contract?",
        "",  # invalid input on purpose
    ]
    for q in tests:
        print(f"\n=== {q!r}")
        print(json.dumps(search_policies(q, top_k=2), indent=2,
                         ensure_ascii=False)[:900])
