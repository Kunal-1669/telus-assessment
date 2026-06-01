"""Generate paraphrases from a fine-tuned T5 or Pegasus model (FP32 or INT8).

Examples:
    # Single input
    python src/infer.py --model checkpoints/t5-small-paraphrase \\
        --text "The quick brown fox jumps over the lazy dog."

    # Multiple paraphrases per input
    python src/infer.py --model checkpoints/t5-small-paraphrase \\
        --text "I forgot my password." --num-return 5

    # Batch from a file (one input per line)
    python src/infer.py --model checkpoints/t5-small-paraphrase \\
        --input-file inputs.txt --output-file out.jsonl

    # Interactive REPL
    python src/infer.py --model checkpoints/t5-small-paraphrase --interactive

    # Load an INT8-quantized checkpoint
    python src/infer.py --model checkpoints/t5-small-paraphrase-int8 --quantized \\
        --text "Hello world."
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def is_t5(model) -> bool:
    """T5 needs a 'paraphrase: ' prefix; Pegasus doesn't."""
    return model.config.model_type == "t5"


def load_model(model_path: str, quantized: bool, device: str):
    """Load FP32 or INT8-quantized model. Quantized always runs on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if quantized:
        import torch.nn as nn
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to("cpu")
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        state = torch.load(Path(model_path) / "pytorch_model_int8.bin", map_location="cpu")
        model.load_state_dict(state)
        device = "cpu"
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def paraphrase(
    text: str,
    model,
    tokenizer,
    device: str,
    num_return: int = 1,
    max_length: int = 128,
    num_beams: int = 5,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 0.95,
):
    prefix = "paraphrase: " if is_t5(model) else ""
    inputs = tokenizer(
        prefix + text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(device)

    gen_kwargs = {
        "max_length": max_length,
        "num_return_sequences": num_return,
    }
    if do_sample:
        gen_kwargs.update(do_sample=True, top_p=top_p, temperature=temperature,
                          num_beams=1)
    else:
        # Beam search: need num_beams >= num_return_sequences
        gen_kwargs.update(num_beams=max(num_beams, num_return),
                          num_beam_groups=1, do_sample=False)

    outputs = model.generate(**inputs, **gen_kwargs)
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]


def run_batch(inputs, model, tokenizer, device, **gen_kwargs):
    results = []
    for text in inputs:
        outs = paraphrase(text, model, tokenizer, device, **gen_kwargs)
        results.append({"source": text, "paraphrases": outs})
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to fine-tuned checkpoint dir")
    p.add_argument("--quantized", action="store_true",
                   help="Load an INT8-quantized checkpoint (from src/quantize.py)")
    p.add_argument("--device", default=None,
                   help="Override device (cpu/mps/cuda). Default: auto.")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Single input sentence")
    src.add_argument("--input-file", help="File with one input per line")
    src.add_argument("--interactive", action="store_true", help="REPL mode")

    p.add_argument("--output-file", help="Write JSONL results (default: stdout)")
    p.add_argument("--num-return", type=int, default=3, help="Paraphrases per input")
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--sample", action="store_true",
                   help="Use nucleus sampling instead of beam search (more diverse)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"Loading model from {args.model} (quantized={args.quantized}, device={device})...",
          file=sys.stderr)
    model, tokenizer, device = load_model(args.model, args.quantized, device)
    print(f"Loaded. model_type={model.config.model_type}", file=sys.stderr)

    gen_kwargs = dict(
        num_return=args.num_return,
        max_length=args.max_length,
        num_beams=args.num_beams,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.interactive:
        print("Interactive mode. Type a sentence (empty line to quit).", file=sys.stderr)
        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not text:
                break
            outs = paraphrase(text, model, tokenizer, device, **gen_kwargs)
            for i, o in enumerate(outs, 1):
                print(f"  {i}. {o}")
        return

    if args.text:
        inputs = [args.text]
    else:
        with open(args.input_file) as f:
            inputs = [ln.strip() for ln in f if ln.strip()]

    results = run_batch(inputs, model, tokenizer, device, **gen_kwargs)

    if args.output_file:
        with open(args.output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(results)} results to {args.output_file}", file=sys.stderr)
    else:
        for r in results:
            print(f"\nSource: {r['source']}")
            for i, o in enumerate(r["paraphrases"], 1):
                print(f"  {i}. {o}")


if __name__ == "__main__":
    main()
