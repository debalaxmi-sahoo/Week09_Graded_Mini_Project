"""MP2 · Mini-RAG — Starter Template
====================================

You'll build a complete RAG pipeline over the Sherlock Holmes corpus in this
file. Fill in every TODO. The reference solution is ~250 lines, but yours can
be shorter or longer — what matters is that it works end-to-end.

Pipeline you're building:
    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations

Run sequence (once you've filled in the TODOs):
    pip install -r requirements.txt
    source .env                 # exports your OpenAI + Qdrant credentials
    python mp2_rag.py ingest    # builds the collection (run once)
    python mp2_rag.py ask       # interactive Q&A loop
    python mp2_rag.py validate  # runs against data/predefined_questions.jsonl

Tip: get the CORE pipeline working FIRST (Steps 1-7 below), THEN come back to
polish and add your 3 questions. Don't try to perfect each step before moving
on — you'll learn more from a rough end-to-end loop than a polished half.
"""
from __future__ import annotations
from dotenv import load_dotenv
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR        = Path(__file__).parent / "corpus"
DATA_DIR          = Path(__file__).parent / "data"
COLLECTION_NAME   = "mp2_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP     = 80    # characters

openai = OpenAI()

load_dotenv()
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    #url="https://bde2dc7f-4cf6-4f51-ae1b-b822abbf7f24.sa-east-1-0.aws.cloud.qdrant.io",
    api_key=os.environ.get("QDRANT_API_KEY"),
    #api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6NjZlZmVkMWMtYTEyNi00Y2U5LWFiMDUtZGQwYjRlMGRmYTBjIn0.ddZHlyszJP8jWt79kk4_OsmJhiL4dhENlo87-rvWvJo",
)


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory.

    Returns a list of dicts, each with: source (filename), title (first line),
    and text (full content).

    TODO:
      - Iterate over every *.txt file in corpus_dir (use Path.glob)
      - For each file, read its text and extract the first non-empty line as title
      - Return the list of doc dicts
    """
    # TODO: your code here
    docs = []

    for file_path in corpus_dir.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        # Find the first non-empty line
        title = ""
        for line in text.splitlines():
            if line.strip():
                title = line.strip()
                break

        docs.append({
            "citation": file_path.name,
            "title": title,
            "text": text,
        })
   
    #raise NotImplementedError("Implement load_corpus")
    return docs


# ─── Step 2: Chunk each document ────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks.

    Each chunk should be a dict with: citation, title, section, text.

    Approach (your choice):
      - Simple: fixed-size windows (split text into N-character chunks with overlap)
      - Smarter: split on paragraph boundaries (\\n\\n), then pack paragraphs
        into chunks up to TARGET_CHUNK_SIZE characters

    The reference solution uses the smarter approach, plus heuristic
    section-header detection (short lines without terminal punctuation).
    Either approach is acceptable.

    TODO:
      - Pick an approach
      - Implement it
      - Return list of chunk dicts
    """
    # TODO: your code here
    # Chunk By Paragraph Strategy
    """Split a document into smaller chunks using paragraph boundaries."""

    text = doc["text"]
    citation = doc["citation"]
    title = doc["title"]

    # Split document into paragraphs
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]
    chunks = []
    current_paragraphs = []
    current_size = 0
    current_section = "General"

    def is_section_header(paragraph: str) -> bool:
        """Heuristic check for a section header."""
        lines = paragraph.splitlines()

        # Header should be short and generally be a single line
        if len(lines) != 1:
            return False

        line = paragraph.strip()

        return (
            len(line) < 80
            and not line.endswith((".", "!", "?", ":"))
        )

    def add_chunk(paragraphs, section):
        if not paragraphs:
            return

        chunk_text = "\n\n".join(paragraphs)

        chunks.append({
            "citation": citation,
            "title": title,
            #"section": section,
            "text": chunk_text,
        })

    for paragraph in paragraphs:

        # Detect section header
        if is_section_header(paragraph):
            # Save the current chunk before changing section
            add_chunk(current_paragraphs, current_section)

            current_paragraphs = []
            current_size = 0

            current_section = paragraph
            continue

        paragraph_size = len(paragraph)

        # If adding this paragraph exceeds the target,
        # save the current chunk first.
        if (
            current_paragraphs
            and current_size + paragraph_size + 2 > TARGET_CHUNK_SIZE
        ):
            add_chunk(current_paragraphs, current_section)

            current_paragraphs = []
            current_size = 0

        current_paragraphs.append(paragraph)
        current_size += paragraph_size + 2

    # Add final chunk
    add_chunk(current_paragraphs, current_section)

    #raise NotImplementedError("Implement chunk_document")

    return chunks
    


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs).

    TODO:
      - Call openai.embeddings.create with EMBEDDING_MODEL and the texts
      - Extract the embedding vectors from the response
    """
    # TODO: your code here
    resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]
 

# ─── Step 4: Set up the Qdrant collection ───────────────────────────────
def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection."""

    print(">>> ENTER setup_collection")

    # Delete any prior version — makes this cell re-runnable
    try:
        qdrant.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing {COLLECTION_NAME!r} collection.")
    except Exception:
        pass  # Collection didn't exist yet

    # Create fresh collection
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=1536,
            distance=Distance.COSINE,
        ),
    )

    info = qdrant.get_collection(COLLECTION_NAME)

    print(f"Created collection {COLLECTION_NAME!r}")
    print(f"  dim:    {info.config.params.vectors.size}")
    print(f"  metric: {info.config.params.vectors.distance}")
    print(f"  points: {info.points_count}")

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant."""
    print("inside ingest_chunks")
    # 1. Get text from each chunk
    print("Number of chunks:", len(chunks))
    chunk_texts = [c["text"] for c in chunks]
    print("Number of texts:", len(chunk_texts))
    # 2. Generate embeddings
    vectors = embed_texts(chunk_texts)
    print("Number of vectors:", len(vectors))
    #print("First vector dimension:", len(vectors[0]))
    # 3. Build Qdrant points
    points = [
        PointStruct(
            id=idx,
            vector=vec,
            payload={
                "title": chunk["title"],
                "citation": chunk["citation"],
                "text": chunk["text"],
            },
        )
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    print("Number of Qdrant points:", len(points))   
    # 4. Upsert into Qdrant
    print("upserting data")
    qdrant.upsert(collection_name=COLLECTION_NAME,points=points)
    info = qdrant.get_collection(COLLECTION_NAME)  
    print(f"Created collection {COLLECTION_NAME!r}")
    print(f"  dim:      {info.config.params.vectors.size}")
    print(f"  metric:   {info.config.params.vectors.distance}")
    print(f"  points:   {info.points_count}")

# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query.

    TODO:
      - Embed the query
      - qdrant.search with the query vector, limit=k
      - Return list of chunk dicts (include score for citations)
    """
    # TODO: your code here
    #- Embed the query
    resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    q_vec = resp.data[0].embedding
   # print("query vector" ,q_vec)
    print("COLLECTION_NAME" ,COLLECTION_NAME)
    #- qdrant.search with the query vector, limit=k
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=q_vec,
        limit=k,
    ).points
    
    print(f"Q: {query!r}")
    print(f"length :{len(results)} ")
    #print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")
    for i, hit in enumerate(results, 1):
        p = hit.payload
        #print(f"  [{i}] score={hit.score:.3f}  {p['animal_id']:8s} ({p['category']})")

    #raise NotImplementedError("Implement retrieve")
    return results

# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result.

    TODO:
      - Call retrieve(question, k=k)
      - Format the retrieved chunks into a context string
        (include "[Source: <title> — <section>]" before each)
      - Call openai.chat.completions.create with SYSTEM_PROMPT and the user message
      - Return dict with: question, answer, citations, latency_ms
    """
    # TODO: your code here

    #Call retrieve(question, k=k)
    hits = retrieve(question, k = k)

    context = "\n\n".join(
        f"[{h.payload['citation']}]\n{h.payload['text']}"
        for h in hits
    )
    context_parts = []
    for chunk in hits:        
        title = chunk.payload.get("title", "")
        citation = chunk.payload.get("citation", "")
        text = chunk.payload.get("text", "")
        #print("title ", title)
        context_parts.append(
            f"[citation: {title} — {citation}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    user_message = f"""Question:{question} Relevant excerpts: {context} """
    # Start latency timer
    start_time = time.perf_counter()
    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content":user_message},
        ],
    )
     # End latency timer
    latency_ms = (time.perf_counter() - start_time) * 1000
    return {
        "question": question,
        "answer":   resp.choices[0].message.content,
        "citation":  [h.payload["citation"] for h in hits],
        "latency_ms": latency_ms,
    }
   # raise NotImplementedError("Implement answer")


# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        print("result citation  ", result["citation"])
        #cited_sources = {cit["citation"] for cit in result["citation"]}
        cited_sources = set(result["citation"])
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print("Ans:",ans_lower)
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['citation']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citation"]:
            #print(f"    - {c['title']} — {c['section']}")
            print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
