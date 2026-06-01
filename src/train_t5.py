"""Fine-tune T5 on paraphrase pairs. Tuned for Apple Silicon (MPS, fp32)."""
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

TASK_PREFIX = "paraphrase: "


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/paraphrases")
    p.add_argument("--model", default="t5-small")
    p.add_argument("--out", default="checkpoints/t5-small-paraphrase")
    p.add_argument("--max-source-len", type=int, default=128)
    p.add_argument("--max-target-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Optional cap on training rows (use for iteration speed)")
    p.add_argument("--max-eval-samples", type=int, default=1000)
    args = p.parse_args()

    device = pick_device()
    print(f"Device: {device}")

    ds = load_from_disk(args.data)
    if args.max_train_samples:
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
    if args.max_eval_samples:
        ds["validation"] = ds["validation"].select(
            range(min(args.max_eval_samples, len(ds["validation"])))
        )
    print(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    def tokenize(batch):
        inputs = [TASK_PREFIX + s for s in batch["source"]]
        model_in = tokenizer(
            inputs,
            max_length=args.max_source_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=args.max_target_len,
            truncation=True,
        )
        model_in["labels"] = labels["input_ids"]
        return model_in

    tokenized = ds.map(
        tokenize,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing",
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=args.max_target_len,
        # fp16 is unstable on MPS — keep fp32
        fp16=False,
        bf16=False,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"Saved model to {args.out}")

    # Quick sanity-check generation
    model.eval()
    sample = ds["validation"][0]
    inp = tokenizer(TASK_PREFIX + sample["source"], return_tensors="pt").to(model.device)
    out = model.generate(**inp, max_length=args.max_target_len, num_beams=4)
    print(f"\nSource: {sample['source']}")
    print(f"Target: {sample['target']}")
    print(f"Generated: {tokenizer.decode(out[0], skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
