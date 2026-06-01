# Custom Paraphrase Generation (CPG)

Paragraph-level paraphrase generation pipeline targeting 200–400 word inputs,
benchmarked against an LLM API baseline. Fine-tuned **T5-small** on
`humarin/chatgpt-paraphrases`, wrapped in a **sentence-chunking pipeline** that
batches generation for low CPU latency.

- [`REPORT.md`](REPORT.md) — final writeup: model choices, metrics, trade-offs,
  error analysis, possible improvements.
- [`EXPERIMENTS.md`](EXPERIMENTS.md) — chronological experiment log: every run,
  every failure, every decision (including the things that didn't work).

## Project layout

```
telus_assessment/
├── src/
│   ├── prepare_data.py          # download + flatten chatgpt-paraphrases dataset
│   ├── train_t5.py              # fine-tune T5 on the flattened pairs
│   ├── train_pegasus.py         # Pegasus fine-tune (deprecated; see REPORT)
│   ├── paragraph_pipeline.py    # CPG: sentence-chunked, batched paraphrase
│   ├── infer.py                 # sentence-level inference CLI
│   ├── quantize.py              # INT8 dynamic quantization + benchmark
│   ├── evaluate.py              # sentence-level BLEU/ROUGE on validation set
│   ├── metrics_paragraph.py     # paragraph-level metrics (BERTScore, BLEU, ROUGE, latency)
│   ├── bench_cpu.py             # CPU latency benchmark vs 800ms target
│   ├── test_passages.py         # bundled test passages (cover letter, legal, medical)
│   └── server.py                # FastAPI POST /paraphrase endpoint
├── notebooks/
│   ├── 01_t5_vs_pegasus.ipynb       # fine-tuned T5 vs off-the-shelf Pegasus
│   └── 02_finetuned_vs_llm.ipynb    # fine-tuned vs Gemini 2.5 Flash
├── fixtures/                    # text dumps of the 3 test passages
├── checkpoints/                 # trained model weights (gitignored)
├── data/                        # cached datasets (gitignored)
├── eval_results/                # benchmark + metrics JSON outputs
└── REPORT.md
```

## Setup

```bash
# Native arm64 Python (Homebrew). Avoid `~/Documents/` if you use iCloud.
/opt/homebrew/bin/python3.12 -m venv ~/.venvs/telus_assessment
ln -s ~/.venvs/telus_assessment venv
source venv/bin/activate

pip install \
  torch transformers datasets accelerate sentencepiece protobuf \
  evaluate sacrebleu rouge-score nltk bert-score \
  jupyter ipykernel google-genai python-dotenv pandas matplotlib tqdm \
  fastapi uvicorn pydantic
python -m ipykernel install --user --name telus-paraphrase --display-name "Python (telus-paraphrase)"
```

> **iCloud warning**: if the project lives under `~/Documents`, macOS iCloud Drive
> can corrupt `venv/` files mid-install (conflict-rename pattern: `file 2.pyc`).
> Keep the venv outside iCloud as shown above.

## API keys

The LLM comparison flows (`metrics_paragraph.py --compare-llm`, notebook 02)
read the Gemini API key from an environment variable. Create `.env` in the
project root with:

```bash
# .env
GEMINI_API_KEY=your-key-here
# optional:
# GEMINI_MODEL=gemini-2.5-flash
```

`.env` is gitignored and excluded from the submission zip. Scripts and
notebooks auto-load it via `python-dotenv`. Alternatively
`export GEMINI_API_KEY=...` in your shell — the code falls back to plain env
vars if no `.env` exists. Get a key at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey).

Other optional `.env` keys (for the FastAPI server):

| Key | Default | Purpose |
|---|---|---|
| `CPG_MODEL` | `checkpoints/t5-small-paraphrase` | Model path served by the API |
| `CPG_DEVICE` | `cpu` | `cpu` / `mps` / `cuda` |
| `CPG_QUANTIZED` | `0` | Set `1` to load an INT8-quantized checkpoint |
| `CPG_NUM_BEAMS` | `5` | Default beam width for `POST /paraphrase` |

## Quickstart

### 1. Prepare the dataset (one-time, ~1 min)
```bash
python src/prepare_data.py --out data/paraphrases
```

### 2. Fine-tune T5-small (~3 hours on Apple M-series, full dataset slice)
```bash
python src/train_t5.py --data data/paraphrases --max-train-samples 100000 --epochs 3
```

### 3. Paraphrase a paragraph
```bash
python src/paragraph_pipeline.py \
  --model checkpoints/t5-small-paraphrase \
  --input-file fixtures/cover_letter.txt
```

### 4. Evaluate
```bash
# Sentence-level on the held-out val set
python src/evaluate.py --model checkpoints/t5-small-paraphrase \
  --data data/paraphrases --max-samples 300

# Paragraph-level on the 3 test passages
python src/metrics_paragraph.py --model checkpoints/t5-small-paraphrase --device cpu

# Optional Gemini comparison (needs GEMINI_API_KEY in .env)
python -m src.metrics_paragraph --model checkpoints/t5-small-paraphrase --device cpu --compare-llm
```

### 5. CPU latency benchmark
```bash
python -m src.bench_cpu --model checkpoints/t5-small-paraphrase \
  --input-file fixtures/cover_letter.txt --repeats 3
```

### 6. Serve as an API
```bash
uvicorn src.server:app --port 8000
# in another terminal:
curl -X POST http://localhost:8000/paraphrase \
  -H "Content-Type: application/json" \
  -d '{"text": "A cover letter is a formal document..."}' | jq
# Or browse http://localhost:8000/docs for Swagger UI
```

### 7. Notebooks
```bash
jupyter lab notebooks/
```

## Reproducibility

Random seed is fixed in `prepare_data.py` (`--seed 42`); the train/val split is
deterministic. Other randomness (beam search is deterministic; sampling not used
by default) is reproducible.
