"""FastAPI inference endpoint for the paragraph paraphrase pipeline.

Run:
    uvicorn src.server:app --reload --port 8000

Endpoints:
    GET  /health         -> liveness + model info
    GET  /passages       -> list bundled test passages
    POST /paraphrase     -> {"text": "..."} -> paraphrase + metadata
    GET  /docs           -> interactive Swagger UI (FastAPI default)
"""
from __future__ import annotations

import os
import time
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()  # CPG_MODEL, CPG_DEVICE, etc. can be set in .env
except ImportError:
    pass

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.paragraph_pipeline import ParagraphParaphraser
from src.test_passages import PASSAGES, list_passages


# Configuration via env vars; sensible defaults for local dev
MODEL_PATH    = os.environ.get("CPG_MODEL",   "checkpoints/t5-small-paraphrase")
DEVICE        = os.environ.get("CPG_DEVICE",  "cpu")
QUANTIZED     = os.environ.get("CPG_QUANTIZED", "0") == "1"
DEFAULT_BEAMS = int(os.environ.get("CPG_NUM_BEAMS", "5"))


app = FastAPI(
    title="Custom Paraphrase Generation API",
    description="Paragraph-level paraphrasing with a fine-tuned T5-small.",
    version="0.1.0",
)


# Lazy global state
_paraphraser: Optional[ParagraphParaphraser] = None


def get_paraphraser() -> ParagraphParaphraser:
    global _paraphraser
    if _paraphraser is None:
        _paraphraser = ParagraphParaphraser(
            MODEL_PATH, device=DEVICE, quantized=QUANTIZED
        )
    return _paraphraser


# ----- Schemas -----

class ParaphraseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000,
                      description="Input paragraph (1-10000 chars)")
    num_beams: int = Field(DEFAULT_BEAMS, ge=1, le=10)
    min_length_ratio: float = Field(0.8, ge=0.0, le=1.5,
                                    description="Min words(out)/words(in) per sentence")
    batch_size: int = Field(16, ge=1, le=64)


class SentenceTrace(BaseModel):
    source: str
    paraphrase: str
    seconds: float


class ParaphraseResponse(BaseModel):
    paraphrase: str
    source_words: int
    paraphrase_words: int
    length_ratio: float
    num_sentences: int
    num_paragraphs: int
    latency_seconds: float
    sentences_per_second: float
    target_800ms_met: bool
    sentences: list[SentenceTrace]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    quantized: bool
    default_num_beams: int


# ----- Endpoints -----

@app.get("/health", response_model=HealthResponse)
def health():
    pp = get_paraphraser()
    return HealthResponse(
        status="ok",
        model=MODEL_PATH,
        device=pp.device,
        quantized=QUANTIZED,
        default_num_beams=DEFAULT_BEAMS,
    )


@app.get("/passages")
def passages():
    """List bundled test passages (id, domain, word_count)."""
    return [
        {"id": pid, "domain": domain, "words": wc}
        for pid, domain, wc in list_passages()
    ]


@app.get("/passages/{passage_id}")
def get_passage(passage_id: str):
    if passage_id not in PASSAGES:
        raise HTTPException(404, f"Unknown passage: {passage_id}")
    return {"id": passage_id, **PASSAGES[passage_id]}


@app.post("/paraphrase", response_model=ParaphraseResponse)
def paraphrase(req: ParaphraseRequest):
    pp = get_paraphraser()
    t0 = time.perf_counter()
    result = pp.paraphrase(
        req.text,
        num_beams=req.num_beams,
        min_length_ratio=req.min_length_ratio,
        batch_size=req.batch_size,
    )
    elapsed = time.perf_counter() - t0
    return ParaphraseResponse(
        paraphrase=result.paraphrase,
        source_words=result.source_words,
        paraphrase_words=result.paraphrase_words,
        length_ratio=result.length_ratio,
        num_sentences=result.num_sentences,
        num_paragraphs=result.num_paragraphs,
        latency_seconds=elapsed,
        sentences_per_second=result.sentences_per_second,
        target_800ms_met=elapsed < 0.8,
        sentences=[
            SentenceTrace(source=s.source, paraphrase=s.paraphrase, seconds=s.seconds)
            for s in result.per_sentence
        ],
    )


# Module-load smoke test for `python -m src.server` (uvicorn is the proper entrypoint)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="127.0.0.1", port=8000, reload=False)
