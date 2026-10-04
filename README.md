# OSHA RAG Assistant

A RAG-based Q&A assistant over OSHA General Industry Standards (29 CFR 1910).
Ask a question in plain language and get an answer grounded in the regulation
text, with the cited sections returned alongside it.

Built to demonstrate production-oriented GenAI/LLM engineering: document
ingestion, structure-aware chunking, vector retrieval, grounded generation,
prompt-injection safeguards, a retrieval evaluation, an API, and a Docker
deployment.

# Why this Project?

At Tenaris, a few times a year, a product would come in with a specific defect or minor damage, and figuring out whether it was acceptable for a given field application meant manually searching through long technical procedures and regulations — often product by product. That kind of problem — an infrequent but real question that requires digging through lengthy documents to answer correctly — is exactly what retrieval-augmented generation is built for. This project applies that same pattern to a different domain, OSHA safety regulations, to build hands-on RAG/LLM experience for the AI Engineer roles I'm now targeting.

## Stack

- **Python 3.11**
- **LangChain** — orchestration (retrieval + prompt chaining)
- **Chroma** — local vector store, persisted to disk
- **OpenAI embeddings + chat model** — swappable via `.env` for an
  open-source stack later (e.g. `sentence-transformers` + a local LLM)
- **FastAPI** — thin API over a pipeline module
- **Docker** — containerization planned, matching the deployment pattern of my other projects
- **pytest**: API integration tests

## How it works

1. **Ingest** (`src/ingest.py`): downloads 29 CFR 1910 from the official eCFR API (XML) and parses it into 204 sections with section id, title, subpart and text.
2. **Chunk** (`src/chunking.py`): one section = one chunk by default. Sections over 500 tokens are split with a token-based splitter (50-token overlap). Result: 2,733 chunks, each keeping its section metadata.
3. **Embed and store** (`src/embed_store.py`): embeds the chunks and persists them in a local Chroma store.
4. **Answer** (`src/rag_chain.py`): embeds the question, retrieves the top 4 chunks, and generates an answer from that context only (`temperature=0`).
5. **Serve** (`src/api.py`): FastAPI endpoints `GET /health` and `POST /query`.

### Prompt-injection safeguards

- The system prompt tells the model to answer only from the provided context and to say so when the context doesn't cover the question.
- Retrieved text is wrapped in `<context>` tags, and the prompt states that this content is data, not instructions.
- A regex heuristic flags obvious override attempts (for example "ignore all previous instructions") and sets `flagged_for_review` in the response. This is a logging aid and is easy to bypass by rewording. The structural defenses above are the real protection.

### Citations

The `sources` field of the response is built directly from the metadata of the retrieved chunks, not from the model's text. It is the reliable record of which sections were used. The model's inline citations can differ from it.

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
- [x] Docker

## Setup (local, step by step)

Requires Python 3.11. Newer versions such as 3.13 fail to install the pinned
`numpy==1.26.4`.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# edit .env and paste your OPENAI_API_KEY

# 4. Download and parse the regulations (downloads 29 CFR 1910 from eCFR)
python -m src.ingest

# 5. Build the vector store (about 2.7k chunks; costs roughly $0.02-0.05)
python -m src.embed_store

# 6. Run the API

uvicorn src.api:app --reload
```

Open `http://localhost:8000/docs` and try `POST /query`, or:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the requirements for eye protection?"}'
```

The response contains `answer`, `sources` (section id, title, URL) and
`flagged_for_review`.

## Docker

With the vector store already built (steps 1-5 above), the API runs in a
container:

```bash
docker compose up --build
```

Then open `http://localhost:8000/docs`. The compose file mounts `./chroma_db`
into the container and passes your `.env` as environment variables. The `.env`
file itself is excluded from the image by `.dockerignore`.

Not yet tested: building the vector store inside the container from a fresh
clone, which should work as

```bash
docker compose run --rm osha-rag-api python -m src.ingest
docker compose run --rm osha-rag-api python -m src.embed_store
```

## Evaluation

`eval/run_eval.py` measures retrieval quality on 20 questions
(`eval/eval_qa.json`) covering 13 different subparts. For each question it
checks whether the expected section appears in the top 4 retrieved chunks.

**Retrieval hit-rate: 100% (20/20).**

The first run scored 95% (19/20). The one miss was a question about general
electrical safety, where I had labeled the expected section as `1910.303`
("General") and the retriever returned `1910.301` ("Introduction") among other
electrical sections. After checking the real section titles in the data, I
corrected the label to `1910.301`. The retrieval itself was not changed.

```bash
python -m eval.run_eval
```

The same script also runs full answer generation on the first 3 questions for
manual review.

## Tests

```bash
python -m pytest
```

`tests/test_api.py` has three integration tests (health check, response shape
of `/query`, injection flag). They call the real pipeline, so they need a built
`chroma_db/` and a valid OpenAI key, and they cost a few cents per run.

## Known gaps

- The evaluation set is small (20 questions). It checks whether the right
  section is retrieved, not whether the generated answer is correct.
- Generated answers are only spot-checked by hand on 3 questions. An automated
  faithfulness check (for example LLM-as-judge) would be the next step.
- The chunk size (500 tokens) and `k=4` are reasonable defaults and have not
  been tuned against the evaluation.
- `GET /health` only confirms the server process is running. It does not check
  the vector store or the OpenAI connection.
- `requirements.txt` pins direct dependencies only. Transitive dependencies
  can still shift between installs.
- `ingest.py` depends on the eCFR XML tag structure (`DIV6` = subpart,
  `DIV8` = section). If parsing returns 0 sections after a schema change,
  inspect `data/raw/1910_raw.xml` and adjust the tag names.
- Chroma prints harmless telemetry warnings (`Failed to send telemetry event`).
- Single-turn only: no conversation memory.

