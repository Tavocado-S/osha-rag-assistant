# OSHA RAG Assistant

A RAG-based Q&A assistant over OSHA General Industry Standards (29 CFR 1910),
built to demonstrate production-oriented GenAI/LLM engineering: document
ingestion, structure-aware chunking, vector retrieval, grounded generation,
prompt-injection safeguards, and a retrieval evaluation harness.

**Why this domain:** "At Tenaris, a few times a year, a product would come in with a specific defect or minor damage, and figuring out whether it was acceptable for a given field application meant manually searching through long technical procedures and regulations — often product by product. That kind of problem — an infrequent but real question that requires digging through lengthy documents to answer correctly — is exactly what retrieval-augmented generation is built for. This project applies that same pattern to a different domain, OSHA safety regulations, to build hands-on RAG/LLM experience for the AI Engineer roles I'm now targeting.."

## Stack

- **LangChain** — orchestration (retrieval + prompt chaining)
- **Chroma** — local vector store, persisted to disk
- **OpenAI embeddings + chat model** — swappable via `.env` for an
  open-source stack later (e.g. `sentence-transformers` + a local LLM)
- **FastAPI** — thin API over a pipeline module
- **Docker** — containerized for deployment parity with the other projects

## Project structure

```
osha-rag-assistant/
├── src/
│   ├── config.py        # env-driven settings
│   ├── ingest.py         # pulls 29 CFR 1910 from the eCFR API -> structured JSON
│   ├── chunking.py       # structure-aware chunking (section = chunk, with metadata)
│   ├── embed_store.py     # embeds chunks, persists Chroma vector store
│   ├── rag_chain.py       # retrieval + prompt template + injection safeguards
│   └── api.py             # FastAPI endpoints (/health, /query)
├── eval/
│   ├── eval_qa.json       # small labeled question -> expected-section set
│   └── run_eval.py        # retrieval hit-rate + answer spot-check
├── tests/
│   └── test_api.py
├── data/                   # raw + structured docs land here (gitignored)
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```
## Progress

- [x] **Ingestion** — pulls OSHA 29 CFR 1910 from the eCFR API, parses it into
      204 structured sections (section id, title, subpart, text). Verified
      working locally.
- [ ] Chunking
- [ ] Vector store + embeddings
- [ ] RAG chain (retrieval + generation + prompt-injection safeguards)
- [ ] FastAPI endpoint
- [ ] Evaluation harness
- [ ] Docker

## Setup (local, step by step)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# edit .env and paste your OPENAI_API_KEY

# 4. Ingest the source documents (downloads 29 CFR 1910 from eCFR)
python -m src.ingest
```

Steps 5 onward (embedding, running the API, evaluation, Docker) will be added
here as those pieces are built and verified.

## Known gaps / honest next steps

- `src/ingest.py`'s XML parsing worked against the live eCFR schema on first
  real run (204 sections parsed) — but if `parse_sections()` ever returns 0
  results after an eCFR schema change, inspect `data/raw/1910_raw.xml` and
  adjust the tag names.
- Everything past ingestion is unbuilt as of this commit.

