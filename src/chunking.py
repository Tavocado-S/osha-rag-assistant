"""
Chunking strategy:

Naive RAG tutorials split raw text every N characters. That's wrong here
because a regulation section is a self-contained unit of meaning — cutting
it mid-paragraph can separate a requirement from the exception that limits
it, which is exactly the kind of error that makes a compliance assistant
dangerous rather than useful.

So the strategy is two-level:
  1. Respect the document's own structure first: one chunk = one section
     (e.g. "1910.212 General requirements for machine guarding"), tagged
     with subpart/section_id/source_url as metadata for citation.
  2. Only if a section exceeds MAX_CHUNK_TOKENS, split it further using
     RecursiveCharacterTextSplitter with overlap, so an oversized section
     doesn't get dropped or truncated — but short/medium sections
     (the majority) stay intact as one semantic unit.

This keeps citations meaningful ("see 1910.212(a)(1)") instead of
"see chunk #47", which matters both for user trust and for prompt-injection
defense (an answer must trace back to a real, checkable section).
"""
import json
from pathlib import Path

import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

STRUCTURED_PATH = Path("data/osha_1910_structured.json")

MAX_CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


def load_sections() -> list[dict]:
    if not STRUCTURED_PATH.exists():
        raise FileNotFoundError(
            f"{STRUCTURED_PATH} not found — run `python -m src.ingest` first."
        )
    return json.loads(STRUCTURED_PATH.read_text(encoding="utf-8"))


def build_documents() -> list[Document]:
    sections = load_sections()
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
        length_function=count_tokens,  # measure in actual tokens, not an approximation
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for sec in sections:
        text = sec["text"]
        metadata = {
            "section_id": sec["section_id"],
            "title": sec["title"],
            "subpart": sec["subpart"],
            "source_url": sec["source_url"],
        }

        if count_tokens(text) <= MAX_CHUNK_TOKENS:
            documents.append(Document(page_content=text, metadata=metadata))
        else:
            sub_chunks = sub_splitter.split_text(text)
            for i, sub_text in enumerate(sub_chunks):
                sub_meta = {**metadata, "sub_chunk": i}
                documents.append(Document(page_content=sub_text, metadata=sub_meta))

    print(f"Built {len(documents)} chunks from {len(sections)} sections.")
    return documents


if __name__ == "__main__":
    docs = build_documents()
    lengths = [count_tokens(d.page_content) for d in docs]
    print(f"Chunk token counts — min: {min(lengths)}, max: {max(lengths)}, "
          f"avg: {sum(lengths) / len(lengths):.0f}")
