"""Vector DB settings that are NOT paths (paths live in the root config.py)."""

COLLECTION_NAME = "policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# 700 keeps every policy section whole (longest section is ~690 chars):
#   - no sentence is cut between two chunks
#   - no chunk is just a heading
# Smaller sizes cut sentences (500 -> 4 cut, 300 -> 23 cut). Before changing
# these, run `python -m pytest tests/test_vector.py` and `python -m vector_db.eval_chunks`.
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
