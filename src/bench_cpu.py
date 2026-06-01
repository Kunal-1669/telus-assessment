"""CPU latency benchmark for the paragraph paraphrase pipeline.

Tests {FP32, INT8} x {beam=5, beam=2, greedy} on CPU against the 800ms target
for inputs under 400 words. Reports median latency, peak memory, throughput,
and writes a JSON summary for the report.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

# Force CPU before any model is loaded
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import psutil  # noqa: E402
import torch  # noqa: E402

from src.paragraph_pipeline import ParagraphParaphraser  # noqa: E402


CONFIGS = [
    # (label,         quantized, num_beams)
    ("FP32 beam=5",   False,     5),
    ("FP32 beam=2",   False,     2),
    ("FP32 greedy",   False,     1),
    ("INT8 beam=5",   True,      5),
    ("INT8 beam=2",   True,      2),
    ("INT8 greedy",   True,      1),
]


def measure(pp: ParagraphParaphraser, text: str, num_beams: int,
            batch_size: int, repeats: int):
    """Run paraphrase `repeats` times, return per-run latencies + peak RSS + last result."""
    # Warmup (kernel JIT, cache warm)
    pp.paraphrase(text, num_beams=num_beams, batch_size=batch_size)

    proc = psutil.Process()
    latencies: list[float] = []
    peak_rss = proc.memory_info().rss
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = pp.paraphrase(text, num_beams=num_beams, batch_size=batch_size)
        latencies.append(time.perf_counter() - t0)
        peak_rss = max(peak_rss, proc.memory_info().rss)
    return latencies, peak_rss, result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Fine-tuned checkpoint dir or HF ID")
    p.add_argument("--input-file", default="cover_letter.txt")
    p.add_argument("--repeats", type=int, default=3,
                   help="Per-config measurement runs (plus 1 warmup)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--target-ms", type=float, default=800.0,
                   help="CPU latency target in milliseconds")
    p.add_argument("--output-json", default="eval_results/cpu_bench.json")
    args = p.parse_args()

    # Sanity check: number of CPU threads. torch will use all by default.
    torch_threads = torch.get_num_threads()
    print(f"PyTorch CPU threads: {torch_threads}")
    print(f"Device: cpu (MPS / CUDA disabled for this run)")

    text = Path(args.input_file).read_text().strip()
    word_count = len(text.split())
    print(f"Input: {args.input_file}  ({word_count} words)\n")

    results: list[dict] = []
    for label, quantized, num_beams in CONFIGS:
        print(f"--- {label} ---")
        print("  loading model...", flush=True)
        t_load = time.perf_counter()
        pp = ParagraphParaphraser(args.model, device="cpu", quantized=quantized)
        print(f"  loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

        print(f"  running {args.repeats} measurement(s)...", flush=True)
        try:
            latencies, peak_rss, result = measure(
                pp, text, num_beams, args.batch_size, args.repeats
            )
        finally:
            del pp
            gc.collect()

        median = statistics.median(latencies)
        out = {
            "config": label,
            "quantized": quantized,
            "num_beams": num_beams,
            "batch_size": args.batch_size,
            "input_words": word_count,
            "median_ms": median * 1000,
            "min_ms": min(latencies) * 1000,
            "max_ms": max(latencies) * 1000,
            "all_runs_ms": [l * 1000 for l in latencies],
            "peak_rss_mb": peak_rss / (1024 ** 2),
            "throughput_words_per_sec": word_count / median,
            "length_ratio": result.length_ratio,
            "paraphrase_words": result.paraphrase_words,
            "target_ms": args.target_ms,
            "target_met": median * 1000 < args.target_ms,
        }
        results.append(out)
        print(f"  median: {out['median_ms']:.0f} ms   "
              f"peak RSS: {out['peak_rss_mb']:.0f} MB   "
              f"words/s: {out['throughput_words_per_sec']:.1f}   "
              f"ratio: {out['length_ratio']:.2f}\n")

    # Pretty table
    sep = "-" * 84
    print("=" * 84)
    print(f"CPU benchmark | input: {word_count} words | target: <{args.target_ms:.0f} ms")
    print(sep)
    print(f"{'Config':<14}  {'Latency':>9}  {'min/max':>13}  "
          f"{'Peak MB':>8}  {'Words/s':>8}  {'Ratio':>5}  {'Target':>6}")
    print(sep)
    for r in results:
        med = f"{r['median_ms']:>6.0f} ms"
        rng = f"{r['min_ms']:>5.0f}/{r['max_ms']:<5.0f}"
        mb  = f"{r['peak_rss_mb']:>6.0f}"
        wps = f"{r['throughput_words_per_sec']:>6.1f}"
        ratio = f"{r['length_ratio']:>4.2f}"
        ok = "PASS" if r["target_met"] else "FAIL"
        print(f"{r['config']:<14}  {med:>9}  {rng:>13}  {mb:>8}  {wps:>8}  {ratio:>5}  {ok:>6}")
    print(sep)

    # JSON dump
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "input_file": str(args.input_file),
            "input_words": word_count,
            "model": args.model,
            "torch_threads": torch_threads,
            "target_ms": args.target_ms,
            "repeats": args.repeats,
            "batch_size": args.batch_size,
            "results": results,
        }, f, indent=2)
    print(f"\nWrote {out_path}")

    # Best config that meets the target
    passing = [r for r in results if r["target_met"]]
    if passing:
        best = min(passing, key=lambda r: r["median_ms"])
        print(f"\nFastest passing config: {best['config']}  ({best['median_ms']:.0f} ms)")
    else:
        closest = min(results, key=lambda r: r["median_ms"])
        print(f"\nNo config met <{args.target_ms:.0f} ms. Closest: "
              f"{closest['config']} ({closest['median_ms']:.0f} ms)")


if __name__ == "__main__":
    main()
