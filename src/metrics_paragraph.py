"""Paragraph-level paraphrase evaluation.

Metrics computed for each passage:
- Length ratio (paraphrase_words / source_words). Target: >=0.80.
- BLEU (source vs paraphrase). LOWER = more rewording, less copy-paste.
- ROUGE-L (source vs paraphrase). Surface overlap.
- BERTScore F1 (source vs paraphrase). Semantic fidelity, HIGHER = closer in meaning.
- Latency (seconds for paraphrase()).

Optional --compare-llm also paraphrases the same passages via Gemini 2.5 Flash
and reports the same metrics + token cost so the report can show a like-for-like
comparison against the fine-tuned model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()  # picks up .env in the project root for API keys
except ImportError:
    pass

import sacrebleu
from rouge_score import rouge_scorer

from src.paragraph_pipeline import ParagraphParaphraser
from src.test_passages import PASSAGES, get


# ----- Metric helpers -----

def _length_ratio(source: str, paraphrase: str) -> float:
    sw = max(1, len(source.split()))
    return len(paraphrase.split()) / sw


def _bleu(source: str, paraphrase: str) -> float:
    # sacrebleu wants list-of-list of refs
    return sacrebleu.sentence_bleu(paraphrase, [source]).score


_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _rouge_l(source: str, paraphrase: str) -> float:
    return _ROUGE.score(source, paraphrase)["rougeL"].fmeasure


_BERT_SCORE = None


def _bert_score(source: str, paraphrase: str) -> float:
    """Lazy-import BERTScore; reuse the same scorer for all calls."""
    global _BERT_SCORE
    if _BERT_SCORE is None:
        from bert_score import BERTScorer
        # roberta-large is the standard for BERTScore but heavy; distilbert is fine for our scale
        _BERT_SCORE = BERTScorer(model_type="distilbert-base-uncased",
                                 lang="en", rescale_with_baseline=False)
    P, R, F1 = _BERT_SCORE.score([paraphrase], [source])
    return float(F1[0])


def compute_metrics(source: str, paraphrase: str) -> dict:
    return {
        "source_words": len(source.split()),
        "paraphrase_words": len(paraphrase.split()),
        "length_ratio": _length_ratio(source, paraphrase),
        "bleu_vs_source": _bleu(source, paraphrase),
        "rougeL_vs_source": _rouge_l(source, paraphrase),
        "bertscore_f1": _bert_score(source, paraphrase),
    }


# ----- LLM paraphrasing (optional) -----

_LLM_PROMPT = (
    "Paraphrase the following passage. Preserve the original meaning "
    "exactly but rewrite using different wording and sentence structures. "
    "Keep the output length within 10% of the original. Return only the "
    "paraphrase, no commentary, no markdown.\n\n"
    "Passage:\n{text}"
)


def paraphrase_gemini(text: str, model: str = "gemini-2.5-flash") -> tuple[str, dict]:
    from google import genai
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    t0 = time.perf_counter()
    resp = client.models.generate_content(
        model=model,
        contents=_LLM_PROMPT.format(text=text),
    )
    elapsed = time.perf_counter() - t0
    paraphrase = (resp.text or "").strip()
    usage = getattr(resp, "usage_metadata", None)
    return paraphrase, {
        "latency_seconds": elapsed,
        "input_tokens":  getattr(usage, "prompt_token_count", 0) if usage else 0,
        "output_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
    }


# ----- Main eval -----

def evaluate_passages(
    passages: dict,
    paraphraser: ParagraphParaphraser,
    num_beams: int = 5,
    min_length_ratio: float = 0.8,
    batch_size: int = 16,
) -> list[dict]:
    results = []
    for pid, p in passages.items():
        source = p["text"]
        t0 = time.perf_counter()
        result = paraphraser.paraphrase(
            source,
            num_beams=num_beams,
            min_length_ratio=min_length_ratio,
            batch_size=batch_size,
        )
        elapsed = time.perf_counter() - t0
        m = compute_metrics(source, result.paraphrase)
        m["passage_id"] = pid
        m["domain"] = p["domain"]
        m["latency_seconds"] = elapsed
        m["paraphrase"] = result.paraphrase
        results.append(m)
    return results


def evaluate_llm(passages: dict, llm_model: str) -> list[dict]:
    results = []
    for pid, p in passages.items():
        source = p["text"]
        try:
            paraphrase, meta = paraphrase_gemini(source, model=llm_model)
        except Exception as e:
            print(f"  LLM call failed for {pid}: {e}")
            paraphrase = ""
            meta = {"latency_seconds": 0.0, "input_tokens": 0, "output_tokens": 0,
                    "error": str(e)}
        m = compute_metrics(source, paraphrase) if paraphrase else {
            "source_words": len(source.split()),
            "paraphrase_words": 0,
            "length_ratio": 0.0,
            "bleu_vs_source": 0.0,
            "rougeL_vs_source": 0.0,
            "bertscore_f1": 0.0,
        }
        m["passage_id"] = pid
        m["domain"] = p["domain"]
        m["latency_seconds"] = meta["latency_seconds"]
        m["paraphrase"] = paraphrase
        m["llm_input_tokens"] = meta.get("input_tokens", 0)
        m["llm_output_tokens"] = meta.get("output_tokens", 0)
        results.append(m)
    return results


def print_table(rows: list[dict], system: str):
    sep = "-" * 96
    print(f"\n{system}")
    print(sep)
    print(f"{'Passage':<25} {'Domain':<8} {'Words':>10} {'Ratio':>6} "
          f"{'BLEU':>6} {'RougeL':>6} {'BERTSc':>7} {'Lat(s)':>7}")
    print(sep)
    for r in rows:
        words = f"{r['source_words']}->{r['paraphrase_words']}"
        print(
            f"{r['passage_id']:<25} {r['domain']:<8} {words:>10} "
            f"{r['length_ratio']:>6.2f} {r['bleu_vs_source']:>6.1f} "
            f"{r['rougeL_vs_source']:>6.3f} {r['bertscore_f1']:>7.3f} "
            f"{r['latency_seconds']:>7.2f}"
        )
    # Averages
    if rows:
        keys = ("length_ratio", "bleu_vs_source", "rougeL_vs_source",
                "bertscore_f1", "latency_seconds")
        avgs = {k: statistics.mean(r[k] for r in rows) for k in keys}
        print(sep)
        print(
            f"{'AVG':<25} {'':<8} {'':>10} "
            f"{avgs['length_ratio']:>6.2f} {avgs['bleu_vs_source']:>6.1f} "
            f"{avgs['rougeL_vs_source']:>6.3f} {avgs['bertscore_f1']:>7.3f} "
            f"{avgs['latency_seconds']:>7.2f}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Fine-tuned checkpoint dir or HF ID")
    p.add_argument("--device", default=None, help="cpu / mps / cuda (default: auto)")
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--min-length-ratio", type=float, default=0.8)
    p.add_argument("--passages", nargs="+", default=None,
                   help="Subset of passage IDs to evaluate (default: all)")
    p.add_argument("--compare-llm", action="store_true",
                   help="Also evaluate via Gemini for comparison")
    p.add_argument("--llm-model", default=None,
                   help="Gemini model name (default: gemini-2.5-flash, or set GEMINI_MODEL in .env)")
    p.add_argument("--output-json", default="eval_results/paragraph_metrics.json")
    args = p.parse_args()

    # Select passages
    if args.passages:
        passages = {pid: get(pid) for pid in args.passages}
    else:
        passages = PASSAGES

    print(f"Evaluating {len(passages)} passage(s): {list(passages)}")

    # Local model
    print(f"\nLoading {args.model} (device={args.device or 'auto'})...")
    pp = ParagraphParaphraser(args.model, device=args.device)
    local_rows = evaluate_passages(
        passages, pp,
        num_beams=args.num_beams,
        min_length_ratio=args.min_length_ratio,
        batch_size=args.batch_size,
    )
    print_table(local_rows, f"=== {args.model} (local) ===")

    output = {
        "model": args.model,
        "device": pp.device,
        "num_beams": args.num_beams,
        "min_length_ratio": args.min_length_ratio,
        "local": local_rows,
    }

    # LLM comparison
    if args.compare_llm:
        if not os.environ.get("GEMINI_API_KEY"):
            print("\nSkipping LLM eval: GEMINI_API_KEY is not set in env / .env.")
        else:
            llm_model = (
                args.llm_model
                or os.environ.get("GEMINI_MODEL")
                or "gemini-2.5-flash"
            )
            print(f"\nRunning LLM eval: gemini / {llm_model}")
            llm_rows = evaluate_llm(passages, llm_model)
            print_table(llm_rows, f"=== {llm_model} (gemini) ===")
            output["llm"] = {
                "provider": "gemini",
                "model": llm_model,
                "results": llm_rows,
            }

    # Save JSON
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
