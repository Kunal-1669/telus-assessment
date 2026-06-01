"""Load humarin/chatgpt-paraphrases, flatten into (source, target) pairs, save to disk."""
import argparse
import ast
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset


def flatten_row(row):
    """Each row has `text` (source) and `paraphrases` (stringified list of 5).
    Emit one (source, target) pair per paraphrase."""
    try:
        paraphrases = ast.literal_eval(row["paraphrases"])
    except (ValueError, SyntaxError):
        paraphrases = []
    return {
        "source": [row["text"]] * len(paraphrases),
        "target": paraphrases,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/paraphrases")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Cap source rows before flattening (for smoke tests)")
    p.add_argument("--val-frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw = load_dataset("humarin/chatgpt-paraphrases", split="train")
    print(f"Raw rows: {len(raw)}")

    if args.max_rows:
        raw = raw.select(range(min(args.max_rows, len(raw))))
        print(f"Capped to: {len(raw)}")

    # Flatten: 1 row -> N rows (one per paraphrase)
    flat = raw.map(
        flatten_row,
        batched=False,
        remove_columns=raw.column_names,
    )
    # Map with non-uniform output sizes -> need to explode manually
    sources, targets = [], []
    for row in flat:
        sources.extend(row["source"])
        targets.extend(row["target"])
    ds = Dataset.from_dict({"source": sources, "target": targets})
    print(f"Flattened pairs: {len(ds)}")

    split = ds.train_test_split(test_size=args.val_frac, seed=args.seed)
    out = DatasetDict({"train": split["train"], "validation": split["test"]})

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save_to_disk(str(out_path))
    print(f"Saved to {out_path}: train={len(out['train'])}, val={len(out['validation'])}")
    print(f"Example: {out['train'][0]}")


if __name__ == "__main__":
    main()
