# OSHA RAG Assistant

A RAG-based Q&A assistant over OSHA General Industry Standards (29 CFR 1910),
built to demonstrate production-oriented GenAI/LLM engineering: document
ingestion, structure-aware chunking, vector retrieval, grounded generation,
prompt-injection safeguards, and a retrieval evaluation harness.

**Why this domain:** built on 15+ years of production/EHS-adjacent
engineering experience (Airbus, Autoliv, Mercedes-Benz, Tenaris Tamsa) —
this is the kind of internal knowledge-base tool that shows up constantly
in enterprise/consulting AI engagements: "let engineers and safety officers
query a large regulatory corpus in natural language instead of searching a
1,000-page PDF."

## Stack

- **LangChain** — orchestration (retrieval + prompt chaining)
- **Chroma** — local vector store, persisted to disk
- **OpenAI embeddings + chat model** — swappable via `.env` for an
  open-source stack later (e.g. `sentence-transformers` + a local LLM)
- **FastAPI** — same pattern as the predictive-maintenance and
  movie-recommender projects: thin API over a pipeline module
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

## Setup (local, step by step)

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# edit .env and paste your OPENAI_API_KEY

# 4. Ingest the source documents (downloads 29 CFR 1910 from eCFR)
python -m src.ingest

# 5. Build the vector store (embeds all chunks — this costs a small
#    amount of OpenAI credit, roughly $0.01-0.05 for the full corpus
#    with text-embedding-3-small)
python -m src.embed_store

# 6. Run the API
uvicorn src.api:app --reload

# 7. Test it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the machine guarding requirements?"}'
```

Or via Docker once steps 4-5 have populated `./chroma_db` locally:

```bash
docker compose up --build
```

## Design notes (the parts worth defending in an interview)

**Chunking strategy** — one chunk = one regulatory section by default
(preserves the unit of meaning: a requirement and its exceptions stay
together), falling back to recursive character splitting with overlap
only for oversized sections. See the docstring in `src/chunking.py` for
the full reasoning — this is the single most interview-relevant design
decision in the project, more than the choice of vector DB.

**Prompt-injection safeguards** — addressed at three layers in
`src/rag_chain.py`: instruction hierarchy in the system prompt, explicit
delimiting of retrieved content as data-not-instructions (defends against
indirect injection via poisoned documents), and a lightweight heuristic
pre-filter for logging obvious override attempts. Documented as
defense-in-depth, not a solved problem — worth saying that explicitly
rather than overclaiming.

**Evaluation** — `eval/run_eval.py` measures retrieval hit-rate (did the
correct section appear in top-k?) separately from generation quality,
since that separation tells you which part of the pipeline to fix when
something's wrong.

## Known gaps / honest next steps

- `src/ingest.py`'s XML parsing targets the eCFR schema as documented,
  but eCFR's tag structure has changed before — if `parse_sections()`
  returns 0 results, inspect `data/raw/1910_raw.xml` and adjust the tag
  names.
- The 5 questions in `eval/eval_qa.json` are a starting scaffold — expand
  this to 20-30 questions covering multiple subparts before treating the
  hit-rate number as meaningful.
- No answer-faithfulness scoring yet (would need an LLM-as-judge pass or
  a framework like RAGAS) — currently just a manual spot-check.
- Single-turn only — no conversation memory/follow-up handling yet.

## Roadmap (1-2 week timeline)

- **Days 1-2:** ingestion + chunking working, inspect chunk quality manually
- **Days 3-4:** vector store + basic retrieval, sanity-check with a few queries
- **Days 5-7:** RAG chain + FastAPI + injection safeguards
- **Days 8-9:** eval harness, expand the QA set, tune chunk size/k based on results
- **Days 10-12:** Docker, README polish, record a short demo (screen capture
  of a query + citation) for the CV/LinkedIn
- **Buffer:** 2-3 days for whatever breaks (eCFR schema, rate limits, etc.)
