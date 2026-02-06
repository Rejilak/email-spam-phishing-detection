import argparse
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .config import TEXT_COL, LABEL_COL, LABELS, BASE_MODEL_NAME, BERT_OUTPUT_DIR
from .preprocess import clean_text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    df = pd.read_csv(args.data).dropna()
    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(clean_text)
    df = df[df[LABEL_COL].isin(LABELS)].copy()

    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for l, i in label2id.items()}
    df["labels"] = df[LABEL_COL].map(label2id).astype(int)

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[LABEL_COL])

    train_ds = Dataset.from_pandas(train_df[[TEXT_COL, "labels"]])
    val_ds = Dataset.from_pandas(val_df[[TEXT_COL, "labels"]])

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    def tok(batch):
        return tokenizer(batch[TEXT_COL], truncation=True, padding="max_length", max_length=args.max_len)

    train_ds = train_ds.map(tok, batched=True)
    val_ds = val_ds.map(tok, batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    BERT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(BERT_OUTPUT_DIR),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=20,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(BERT_OUTPUT_DIR))
    tokenizer.save_pretrained(str(BERT_OUTPUT_DIR))
    (BERT_OUTPUT_DIR / "labels.txt").write_text("\n".join(LABELS), encoding="utf-8")
    print(f"Saved fine-tuned LLM model to: {BERT_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
