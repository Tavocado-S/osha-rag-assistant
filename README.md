# OSHA RAG Assistant

A RAG-based Q&A assistant over OSHA General Industry Standards (29 CFR 1910),
built to demonstrate production-oriented GenAI/LLM engineering: document
ingestion, structure-aware chunking, vector retrieval, grounded generation,
prompt-injection safeguards, and a retrieval evaluation harness.

**Why this domain:** "At Tenaris, a few times a year, a product would come in with a specific defect or minor damage, and figuring out whether it was acceptable for a given field application meant manually searching through long technical procedures and regulations — often product by product. That kind of problem — an infrequent but real question that requires digging through lengthy documents to answer correctly — is exactly what retrieval-augmented generation is built for. This project applies that same pattern to a different domain, OSHA safety regulations, to build hands-on RAG/LLM experience for the AI Engineer roles I'm now targeting."

## Stack

- **LangChain** — orchestration (retrieval + prompt chaining)
- **Chroma** — local vector store, persisted to disk
- **OpenAI embeddings + chat model** — swappable via `.env` for an
  open-source stack later (e.g. `sentence-transformers` + a local LLM)
- **FastAPI** — thin API over a pipeline module
- **Docker** — containerization planned, matching the deployment pattern of my other projects

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
- [x] **Chunking** — structure-aware splitting (one section = one chunk by
      default), token-based sizing. 2,733 chunks from 204 sections.
- [x] **Vector store + embeddings** — all chunks embedded with
      text-embedding-3-small, persisted to a local Chroma store.
- [x] **RAG chain** — retrieval + grounded generation + prompt-injection
      safeguards, verified against normal, out-of-scope, and injection-
      attempt queries.
- [x] **FastAPI endpoint** — `/health` and `/query`, tested live via the
      auto-generated Swagger UI.
- [x] **Evaluation harness** — retrieval hit-rate measured on a 20-question
      set spanning 13 subparts: 100%. See Evaluation section below.
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

# 5. Build the vector store (embeds all chunks — costs roughly $0.02-0.05
#    with text-embedding-3-small)
python -m src.embed_store

# 6. Run the API
uvicorn src.api:app --reload

# 7. Test it (in a separate terminal, or via http://localhost:8000/docs)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the machine guarding requirements?"}'

# 8. Run the evaluation harness
python -m eval.run_eval
```

Docker setup will be added here once containerization is complete.

## Evaluation

Retrieval quality was measured with `eval/run_eval.py` against a 20-question
test set (`eval/eval_qa.json`) spanning 13 different OSHA subparts — not
just the topics covered during development, to avoid testing only on
"easy" cases.

**Retrieval hit-rate: 100% (20/20)** — the correct section appeared in the
top-4 retrieved chunks for every question. One initial miss (95%, 19/20)
was traced to an imprecise expected-section label in the test set itself
(confirmed by checking the real section titles in the data), not a
retrieval failure — corrected after verification.

A separate manual spot-check runs full generation (not just retrieval) on
a subset of 3 questions, to review answer quality and citation behavior
without the added API cost of generating all 20.

## Known gaps / honest next steps

- `src/ingest.py`'s XML parsing worked against the live eCFR schema on first
  real run (204 sections parsed) — but if `parse_sections()` ever returns 0
  results after an eCFR schema change, inspect `data/raw/1910_raw.xml` and
  adjust the tag names.
- The AI's generated answer doesn't always explicitly cite every section it
  retrieved — the `sources` field in the response is the reliable source of
  truth (pulled directly from retrieved document metadata), not the AI's
  inline prose citations.
- The eval set (20 questions) is still a relatively small sample for
  statistical confidence — expanding further would strengthen the
  retrieval hit-rate as evidence.
- The manual answer spot-check only reviews full generation on 3 of the
  20 questions (API cost trade-off) — a more rigorous setup would use an
  automated faithfulness check (e.g. LLM-as-judge) across the full set.
- No Docker deployment yet.

