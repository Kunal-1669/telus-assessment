"""Compute BLEU + ROUGE-L on the validation set for a fine-tuned paraphrase model.

Each source in the val set has one reference target. For a fair single-reference
eval we generate one paraphrase per source and score against the reference.

Usage:
    python src/evaluate.py --model checkpoints/t5-small-paraphrase --data data/paraphrases
    python src/evaluate.py --model checkpoints/pegasus-paraphrase --data data/paraphrases \\
        --max-samples 500
"""
import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import sacrebleu
from rouge_score import rouge_scorer


def pick_device(prefer=None):
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_t5(model):
    return model.config.model_type == "t5"


@torch.no_grad()
def generate_batch(texts, model, tokenizer, device, max_length, num_beams):
    prefix = "paraphrase: " if is_t5(model) else ""
    inputs = tokenizer(
        [prefix + t for t in texts],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        num_return_sequences=1,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data", default="data/paraphrases")
    p.add_argument("--split", default="validation")
    p.add_argument("--max-samples", type=int, default=500,
                   help="Cap eval samples (full val set is slow on MPS)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-beams", type=int, default=4)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--output", default=None,
                   help="Where to save per-example results JSONL. "
                        "Default: <model>/eval_<split>.jsonl")
    args = p.parse_args()

    device = pick_device(args.device)
    model_path = Path(args.model)
    is_local = model_path.exists() and model_path.is_dir()

    if args.output:
        out_path = Path(args.output)
    elif is_local:
        out_path = model_path / f"eval_{args.split}.jsonl"
    else:
        safe = args.model.replace("/", "__")
        out_path = Path("eval_results") / safe / f"eval_{args.split}.jsonl"
    summary_path = out_path.with_suffix(".summary.json")

    print(f"Loading {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    model.eval()

    ds = load_from_disk(args.data)[args.split]
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"Evaluating on {len(ds)} examples (model_type={model.config.model_type})")

    sources = ds["source"]
    references = ds["target"]
    predictions = []

    t0 = time.perf_counter()
    for i in range(0, len(sources), args.batch_size):
        batch = sources[i : i + args.batch_size]
        preds = generate_batch(batch, model, tokenizer, device,
                               args.max_length, args.num_beams)
        predictions.extend(preds)
        if (i // args.batch_size) % 10 == 0:
            done = i + len(batch)
            rate = done / (time.perf_counter() - t0)
            eta = (len(sources) - done) / max(rate, 1e-9)
            print(f"  {done}/{len(sources)}  ({rate:.1f} ex/s, ETA {eta:.0f}s)")

    elapsed = time.perf_counter() - t0
    print(f"Generation done in {elapsed:.1f}s ({len(sources)/elapsed:.1f} ex/s)")

    # Corpus-level BLEU (sacrebleu expects list-of-lists for refs)
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    # ROUGE-L per example, then average
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    per_example = []
    for src, ref, pred in zip(sources, references, predictions):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
        per_example.append({
            "source": src,
            "reference": ref,
            "prediction": pred,
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        })

    summary = {
        "model": str(model_path),
        "model_type": model.config.model_type,
        "split": args.split,
        "num_samples": len(sources),
        "num_beams": args.num_beams,
        "bleu": bleu.score,
        "rouge1": sum(r1) / len(r1),
        "rouge2": sum(r2) / len(r2),
        "rougeL": sum(rl) / len(rl),
        "generation_seconds": elapsed,
        "examples_per_second": len(sources) / elapsed,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in per_example:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Summary ---")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print(f"\nPer-example results: {out_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
