# vector_db - `search_policies` tool (ChromaDB)

Semantic search over the TeleNova policy documents in `data/policies/`.
All commands run from the repo root.

## Setup
```bash
python -m pip install -r requirements.txt
python -m vector_db.build_vectordb   # once, and after any policy change
python -m vector_db.search           # quick self-test
python -m pytest tests/test_vector.py -v
```
The first run downloads the `all-MiniLM-L6-v2` embedding model (about 80 MB).

## Use from the agent
```python
from vector_db.search import search_policies
result = search_policies("What offers exist for long-term customers?", top_k=3)
```

## Return format (shared tool contract)
```json
{"status": "ok", "tool": "search_policies",
 "data": {"query": "...", "results": [
   {"text": "...", "source_file": "retention_offers.txt",
    "chunk_id": "retention_offers_4", "topic": "...", "distance": 0.21}]},
 "error": null}
```
Failure (never raises): `{"status": "error", "tool": "search_policies", "data": null, "error": {"type": "...", "message": "...", "missing_fields": []}}`

Error types: `validation_error`, `not_found`, `connection_error`, `internal_error`.
`distance` is cosine distance: closer to 0 means more similar.

## Files
- `build_vectordb.py` builds the store (chunks + embeddings + metadata `source_file`, `chunk_id`, `topic`).
- `search.py` the tool. `settings.py` model / chunk size. Paths come from the root `config.py`.
- `eval_chunks.py` compares chunk sizes on a question set.
