"""
FastAPI layer — reuses the same pattern as your other projects
(movie-recommender, predictive maintenance): thin API wrapping a
pipeline module, so the RAG logic in rag_chain.py stays testable
independent of the web framework.

Run locally:
    uvicorn src.api:app --reload

Run via Docker:
    docker compose up --build
"""
from fastapi import FastAPI
from pydantic import BaseModel

from src.rag_chain import answer_question

app = FastAPI(
    title="OSHA RAG Assistant",
    description="RAG-based Q&A over OSHA 29 CFR 1910 General Industry Standards",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    question: str


class SourceRef(BaseModel):
    section_id: str | None
    title: str | None
    source_url: str | None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    flagged_for_review: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = answer_question(request.question)
    return result
