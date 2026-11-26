"""
Build the ChromaDB vector store from the policy documents.
Run from the repo root (once, and again whenever a policy file changes):

    python -m vector_db.build_vectordb
"""
import json
import shutil

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from config import CHROMA_STORE_DIR, POLICIES_DIR, SHARED_SCHEMA_PATH
from vector_db.settings import (CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
                                EMBEDDING_MODEL)


def load_documents() -> dict:
    """Read every .txt policy file. Returns {file_name: text}."""
    files = sorted(POLICIES_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {POLICIES_DIR}")
    return {f.name: f.read_text(encoding="utf-8") for f in files}


def load_topics() -> dict:
    """{file_name: topic} from data/shared_schema.json (policy_documents)."""
    schema = json.loads(SHARED_SCHEMA_PATH.read_text(encoding="utf-8"))
    return {d["file"]: d["topic"] for d in schema["policy_documents"]}


HEADING_MAX_CHARS = 80


def _is_heading(line: str) -> bool:
    """A short single line without a final period, e.g. 'Upgrading a Contract'."""
    line = line.strip()
    return 0 < len(line) <= HEADING_MAX_CHARS and not line.endswith((".", ":", "?"))


def split_sections(text: str):
    """Split a policy file into (heading, body) sections.

    Sections are separated by blank lines. A heading is either the first line of
    a block ('Heading\nBody...') or a block on its own line (then it belongs to
    the next block). A heading is never separated from its body.
    """
    sections, pending = [], ""
    for block in (b.strip() for b in text.split("\n\n")):
        if not block:
            continue
        lines = block.split("\n")
        if len(lines) == 1 and _is_heading(block):      # heading alone -> keep for next block
            pending = f"{pending}\n{block}".strip()
        elif len(lines) > 1 and _is_heading(lines[0]):  # 'Heading\nBody'
            heading = f"{pending}\n{lines[0]}".strip()
            sections.append((heading, "\n".join(lines[1:]).strip()))
            pending = ""
        else:                                           # plain paragraph / FAQ item
            sections.append((pending, block))
            pending = ""
    if pending:
        sections.append((pending, ""))
    return sections


def split_text(text: str, size: int, overlap: int):
    """Chunks of at most ~`size` chars. Long sections are split, and the heading
    is repeated at the top of every piece so no chunk is just a title."""
    chunks = []
    for heading, body in split_sections(text):
        prefix = f"{heading}\n" if heading else ""
        whole = f"{prefix}{body}".strip()
        if len(whole) <= size or not body:
            chunks.append(whole)
            continue
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(size - len(prefix), 100), chunk_overlap=overlap
        )
        chunks += [f"{prefix}{piece}" for piece in splitter.split_text(body)]
    return chunks


def chunk_documents(documents: dict, topics: dict,
                    size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split each document into chunks. Returns (ids, texts, metadatas)."""
    ids, texts, metadatas = [], [], []
    for file_name, text in documents.items():
        for i, chunk in enumerate(split_text(text, size, overlap)):
            chunk_id = f"{file_name.removesuffix('.txt')}_{i}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({
                "source_file": file_name,
                "chunk_id": chunk_id,
                "topic": topics.get(file_name, ""),
            })
    return ids, texts, metadatas


def main():
    documents = load_documents()
    print(f"Loaded {len(documents)} documents from {POLICIES_DIR}")

    ids, texts, metadatas = chunk_documents(documents, load_topics())
    print(f"Created {len(texts)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

    print(f"Embedding with {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()

    # Rebuild from scratch so re-running never creates duplicates.
    if CHROMA_STORE_DIR.exists():
        shutil.rmtree(CHROMA_STORE_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_STORE_DIR))
    collection = client.create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
    collection.add(ids=ids, documents=texts,
                   embeddings=embeddings, metadatas=metadatas)

    print(f"Stored {collection.count()} chunks in {CHROMA_STORE_DIR}")
    for name in documents:
        n = sum(1 for m in metadatas if m["source_file"] == name)
        print(f"  {name}: {n} chunks")


if __name__ == "__main__":
    main()
