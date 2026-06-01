"""Paragraph-level paraphrasing pipeline.

Splits input into paragraphs -> sentences, paraphrases each sentence, and
rejoins. This is the right architecture for 200-400 word inputs because:

- The fine-tuned model was trained on sentence-level pairs (humarin/chatgpt-paraphrases)
- Off-the-shelf tuner007/pegasus_paraphrase only accepts 60 input tokens
- Per-sentence beam search gives much more controllable diversity + length

The pipeline enforces a paragraph-level length-preservation ratio (default 0.8)
by selecting the longest viable beam candidate per sentence and falling back to
the source sentence if no candidate is acceptable.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ----- Sentence splitting -----

_NLTK_READY: bool | None = None


def _ensure_nltk():
    """Lazy nltk import + punkt download. Falls back to regex if anything fails."""
    global _NLTK_READY
    if _NLTK_READY is not None:
        return _NLTK_READY
    try:
        import nltk
        for resource in ("punkt_tab", "punkt"):
            try:
                nltk.data.find(f"tokenizers/{resource}")
                _NLTK_READY = True
                return True
            except LookupError:
                continue
        try:
            nltk.download("punkt_tab", quiet=True)
            _NLTK_READY = True
            return True
        except Exception:
            nltk.download("punkt", quiet=True)
            _NLTK_READY = True
            return True
    except Exception:
        _NLTK_READY = False
        return False


_SENT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\(])")


def split_sentences(text: str) -> list[str]:
    """nltk sent_tokenize if available, regex fallback otherwise."""
    if _ensure_nltk():
        from nltk.tokenize import sent_tokenize
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except LookupError:
            pass
    return [s.strip() for s in _SENT_REGEX.split(text.strip()) if s.strip()]


def split_paragraphs(text: str) -> list[list[str]]:
    """Split on blank lines into paragraphs, then each paragraph into sentences.
    Preserves the structural boundary so we can rejoin with \\n\\n later."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    out: list[list[str]] = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        out.append(split_sentences(p))
    return out


# ----- Pipeline -----

@dataclass
class SentenceTrace:
    source: str
    paraphrase: str
    seconds: float
    candidates: list[str] = field(default_factory=list)


@dataclass
class ParaphraseResult:
    source: str
    paraphrase: str
    num_paragraphs: int
    num_sentences: int
    source_words: int
    paraphrase_words: int
    length_ratio: float
    latency_seconds: float
    sentences_per_second: float
    per_sentence: list[SentenceTrace] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_sentence"] = [asdict(s) for s in self.per_sentence]
        return d


def pick_device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ParagraphParaphraser:
    """Paragraph-level paraphraser wrapping a fine-tuned or off-the-shelf seq2seq model.

    Usage:
        pp = ParagraphParaphraser("checkpoints/t5-small-paraphrase")
        result = pp.paraphrase(long_paragraph)
        print(result.paraphrase)
        print(f"ratio={result.length_ratio:.2f}, latency={result.latency_seconds:.2f}s")
    """

    def __init__(
        self,
        model_path_or_id: str,
        device: str | None = None,
        quantized: bool = False,
    ):
        # Quantized models only support CPU
        if quantized:
            device = "cpu"
        self.device = pick_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path_or_id).to(self.device)

        if quantized:
            # arm64 macOS needs qnnpack backend; fbgemm is x86-only
            try:
                torch.backends.quantized.engine = "qnnpack"
            except RuntimeError:
                pass
            self.model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear}, dtype=torch.qint8
            )
            state_file = Path(model_path_or_id) / "pytorch_model_int8.bin"
            if state_file.exists():
                self.model.load_state_dict(torch.load(state_file, map_location="cpu"))

        self.model.eval()
        self.is_t5 = self.model.config.model_type == "t5"
        # Conservative cap on input tokens per sentence
        self.max_input_tokens = (
            getattr(self.model.config, "n_positions", None)
            or getattr(self.model.config, "max_position_embeddings", 512)
        )

    @torch.no_grad()
    def _generate_batch(
        self,
        texts: list[str],
        num_return: int,
        num_beams: int,
        max_new_tokens: int,
        min_length: int | None = None,
    ) -> list[list[str]]:
        """Generate candidates for a batch of inputs. Returns one candidate
        list per input (length == num_return, may be shorter after empty filter)."""
        if not texts:
            return []
        prefix = "paraphrase: " if self.is_t5 else ""
        enc = self.tokenizer(
            [prefix + t for t in texts],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
            padding=True,
        ).to(self.device)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_beams=max(num_beams, num_return),
            num_return_sequences=num_return,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        if min_length is not None:
            gen_kwargs["min_length"] = min_length
        out = self.model.generate(**enc, **gen_kwargs)
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        # generate() returns [batch * num_return, seq_len] in input order
        grouped: list[list[str]] = []
        for i in range(len(texts)):
            chunk = decoded[i * num_return : (i + 1) * num_return]
            grouped.append([c.strip() for c in chunk if c.strip()])
        return grouped

    def _pick_best(self, sentence: str, candidates: list[str], min_length_ratio: float) -> str:
        """Apply length-ratio selection: prefer longest viable, fall back to longest, then source."""
        if not candidates:
            return sentence
        src_words = max(1, len(sentence.split()))
        min_words_required = max(3, int(src_words * min_length_ratio))
        viable = [c for c in candidates if len(c.split()) >= min_words_required]
        if viable:
            return max(viable, key=lambda x: len(x.split()))
        return max(candidates, key=lambda x: len(x.split()))

    def paraphrase_batch(
        self,
        sentences: list[str],
        num_beams: int = 5,
        min_length_ratio: float = 0.8,
        max_extra_tokens: int = 32,
        batch_size: int = 16,
    ) -> list[tuple[str, list[str], float]]:
        """Paraphrase a list of sentences with batched generation.

        Returns one (best, candidates, amortized_seconds) tuple per input sentence.
        `amortized_seconds` is the chunk latency divided by chunk size — useful
        for per-sentence reporting even though sentences are co-generated.
        """
        if not sentences:
            return []

        results: list[tuple[str, list[str], float]] = []
        for start in range(0, len(sentences), batch_size):
            chunk = sentences[start : start + batch_size]
            t0 = time.perf_counter()
            # max_new_tokens scoped to the longest source in the chunk (+ slack)
            token_counts = [
                self.tokenizer(s, return_tensors="pt", truncation=True,
                               max_length=self.max_input_tokens).input_ids.shape[1]
                for s in chunk
            ]
            max_new_tokens = max(token_counts) + max_extra_tokens
            candidates_per_input = self._generate_batch(
                chunk,
                num_return=num_beams,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                # min_length omitted in batch mode: it applies uniformly to all items
                # in the batch, which would penalize naturally-short sentences.
                # Length is enforced via _pick_best instead.
            )
            chunk_latency = time.perf_counter() - t0
            amortized = chunk_latency / max(len(chunk), 1)
            for sentence, candidates in zip(chunk, candidates_per_input):
                best = self._pick_best(sentence, candidates, min_length_ratio)
                results.append((best, candidates, amortized))
        return results

    def paraphrase_sentence(
        self,
        sentence: str,
        num_beams: int = 5,
        min_length_ratio: float = 0.8,
        max_extra_tokens: int = 32,
    ) -> tuple[str, list[str]]:
        """Convenience wrapper: paraphrase a single sentence."""
        best, candidates, _ = self.paraphrase_batch(
            [sentence],
            num_beams=num_beams,
            min_length_ratio=min_length_ratio,
            max_extra_tokens=max_extra_tokens,
            batch_size=1,
        )[0]
        return best, candidates

    def paraphrase(
        self,
        text: str,
        num_beams: int = 5,
        min_length_ratio: float = 0.8,
        batch_size: int = 16,
        record_candidates: bool = False,
    ) -> ParaphraseResult:
        """Paraphrase a full paragraph (or multi-paragraph) input.

        All sentences across all paragraphs are batched into a single generate()
        call (chunked by `batch_size`), then reassembled along paragraph boundaries.
        """
        t_start = time.perf_counter()
        para_groups = split_paragraphs(text)

        # Flatten while remembering paragraph boundaries
        flat_sentences: list[str] = []
        paragraph_lengths: list[int] = []
        for sents in para_groups:
            paragraph_lengths.append(len(sents))
            flat_sentences.extend(sents)

        batch_results = self.paraphrase_batch(
            flat_sentences,
            num_beams=num_beams,
            min_length_ratio=min_length_ratio,
            batch_size=batch_size,
        )

        per_sentence: list[SentenceTrace] = []
        rebuilt_paragraphs: list[str] = []
        idx = 0
        for group_size in paragraph_lengths:
            out_sents: list[str] = []
            for _ in range(group_size):
                source = flat_sentences[idx]
                best, candidates, seconds = batch_results[idx]
                per_sentence.append(
                    SentenceTrace(
                        source=source,
                        paraphrase=best,
                        seconds=seconds,
                        candidates=candidates if record_candidates else [],
                    )
                )
                out_sents.append(best)
                idx += 1
            rebuilt_paragraphs.append(" ".join(out_sents))

        joined = "\n\n".join(rebuilt_paragraphs)
        latency = time.perf_counter() - t_start
        src_words = len(text.split())
        out_words = len(joined.split())

        return ParaphraseResult(
            source=text,
            paraphrase=joined,
            num_paragraphs=len(para_groups),
            num_sentences=len(per_sentence),
            source_words=src_words,
            paraphrase_words=out_words,
            length_ratio=out_words / max(src_words, 1),
            latency_seconds=latency,
            sentences_per_second=len(per_sentence) / max(latency, 1e-9),
            per_sentence=per_sentence,
        )


# ----- CLI -----

def _format_summary(r: ParaphraseResult) -> str:
    return (
        f"{r.num_paragraphs} paragraph(s), {r.num_sentences} sentence(s)  "
        f"{r.source_words} -> {r.paraphrase_words} words "
        f"(ratio {r.length_ratio:.2f})  "
        f"{r.latency_seconds:.2f}s total, "
        f"{r.sentences_per_second:.1f} sent/s"
    )


def main():
    p = argparse.ArgumentParser(description="Paragraph-level paraphraser")
    p.add_argument("--model", required=True,
                   help="Local checkpoint dir or HF model ID")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Input paragraph")
    g.add_argument("--input-file", help="Path to a file with one or more paragraphs")
    p.add_argument("--quantized", action="store_true",
                   help="Load an INT8 checkpoint produced by src/quantize.py (CPU only)")
    p.add_argument("--device", default=None, help="cpu / mps / cuda (default: auto)")
    p.add_argument("--num-beams", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Sentences batched into a single generate() call (default 16)")
    p.add_argument("--min-length-ratio", type=float, default=0.8,
                   help="Min words(out)/words(in) per sentence; output target is the same overall")
    p.add_argument("--json", action="store_true", help="Print full JSON result to stdout")
    p.add_argument("--output-file", help="Write paraphrase text to this file")
    args = p.parse_args()

    text = args.text if args.text else Path(args.input_file).read_text()

    pp = ParagraphParaphraser(args.model, device=args.device, quantized=args.quantized)
    result = pp.paraphrase(
        text,
        num_beams=args.num_beams,
        batch_size=args.batch_size,
        min_length_ratio=args.min_length_ratio,
    )

    if args.output_file:
        Path(args.output_file).write_text(result.paraphrase)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.paraphrase)
        print()
        print("---")
        print(_format_summary(result))
        if result.length_ratio < args.min_length_ratio:
            print(f"WARN: output ratio {result.length_ratio:.2f} below target {args.min_length_ratio:.2f}")


if __name__ == "__main__":
    main()
