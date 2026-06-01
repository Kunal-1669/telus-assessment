"""Post-training INT8 dynamic quantization for fine-tuned seq2seq models.

Dynamic quantization converts nn.Linear weights to int8 and quantizes activations
on the fly. Works on CPU, no calibration data needed. Typical results for T5/Pegasus:
~4x smaller, 1.5-3x faster on CPU, minor BLEU drop.
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def model_size_mb(model: nn.Module) -> float:
    """Sum of parameter and buffer sizes in MB."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / (1024 ** 2)


def disk_size_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 2)


def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    """INT8 dynamic quantization of all nn.Linear layers."""
    return torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )


@torch.no_grad()
def benchmark_generate(model, tokenizer, prompts, max_length=128, num_beams=4, warmup=1):
    """Average per-prompt generation latency in seconds."""
    model.eval()
    # Warmup (kernel caching, lazy init)
    for _ in range(warmup):
        inp = tokenizer(prompts[0], return_tensors="pt", truncation=True, max_length=128)
        model.generate(**inp, max_length=max_length, num_beams=num_beams)

    t0 = time.perf_counter()
    outputs = []
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt", truncation=True, max_length=128)
        out = model.generate(**inp, max_length=max_length, num_beams=num_beams)
        outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))
    elapsed = time.perf_counter() - t0
    return elapsed / len(prompts), outputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to fine-tuned checkpoint dir")
    p.add_argument("--out", required=True, help="Where to save quantized model")
    p.add_argument("--task-prefix", default="",
                   help="Set to 'paraphrase: ' for T5; leave empty for Pegasus")
    p.add_argument("--num-prompts", type=int, default=5,
                   help="How many prompts to benchmark over")
    args = p.parse_args()

    model_path = Path(args.model)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_path} on CPU (quantization runs on CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_fp32 = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cpu")

    fp32_mem = model_size_mb(model_fp32)
    fp32_disk = disk_size_mb(model_path)
    print(f"FP32: in-memory={fp32_mem:.1f} MB, on-disk={fp32_disk:.1f} MB")

    print("Quantizing (dynamic INT8, nn.Linear)...")
    model_int8 = quantize_dynamic_int8(model_fp32)
    int8_mem = model_size_mb(model_int8)
    print(f"INT8 in-memory: {int8_mem:.1f} MB ({fp32_mem / max(int8_mem, 1e-9):.2f}x smaller)")

    # Save: quantized models can't use save_pretrained reliably (HF format expects
    # standard dtype tensors). Use torch.save for the state dict + keep tokenizer + config.
    torch.save(model_int8.state_dict(), out_path / "pytorch_model_int8.bin")
    model_fp32.config.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    int8_disk = disk_size_mb(out_path)
    print(f"Saved to {out_path}: on-disk={int8_disk:.1f} MB ({fp32_disk / max(int8_disk, 1e-9):.2f}x smaller)")

    # Build benchmark prompts
    prompts_raw = [
        "The quick brown fox jumps over the lazy dog.",
        "I cannot remember my password and the recovery email is no longer active.",
        "Climate change is one of the most pressing challenges of our time.",
        "She walked into the room and noticed something was different.",
        "Machine learning models require large amounts of training data.",
    ][: args.num_prompts]
    prompts = [args.task_prefix + p for p in prompts_raw]

    print("\nBenchmarking FP32 generation...")
    t_fp32, out_fp32 = benchmark_generate(model_fp32, tokenizer, prompts)
    print(f"FP32 avg latency: {t_fp32*1000:.0f} ms/prompt")

    print("Benchmarking INT8 generation...")
    t_int8, out_int8 = benchmark_generate(model_int8, tokenizer, prompts)
    print(f"INT8 avg latency: {t_int8*1000:.0f} ms/prompt ({t_fp32 / max(t_int8, 1e-9):.2f}x speedup)")

    print("\n--- Sample outputs ---")
    for src, a, b in zip(prompts_raw, out_fp32, out_int8):
        print(f"\nSource: {src}")
        print(f"  FP32: {a}")
        print(f"  INT8: {b}")

    print("\n--- Summary ---")
    print(f"Size:    {fp32_mem:.1f} -> {int8_mem:.1f} MB  ({fp32_mem / max(int8_mem, 1e-9):.2f}x)")
    print(f"Latency: {t_fp32*1000:.0f} -> {t_int8*1000:.0f} ms  ({t_fp32 / max(t_int8, 1e-9):.2f}x)")


def load_quantized(model_dir: str | Path):
    """Reload a quantized checkpoint saved by this module.

    Usage:
        model, tokenizer = load_quantized("checkpoints/t5-small-paraphrase-int8")
        out = model.generate(**tokenizer("paraphrase: hello", return_tensors="pt"))
    """
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cpu")
    model = quantize_dynamic_int8(model)
    state = torch.load(model_dir / "pytorch_model_int8.bin", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model, tokenizer


if __name__ == "__main__":
    main()
