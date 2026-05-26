# 2_train_teacher_crf.py
import os
import torch
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from config import MODEL_NAME, PROCESSED_DATA_DIR, LABEL2ID, ID2LABEL
from crf_model import BanglaBertCRF

# Load NER evaluation metric
metric = evaluate.load("seqeval")


def compute_metrics(p):
    """
    CRITICAL DIFFERENCE from standard BERT training:
    The CRF layer already outputs final class indices (Viterbi decoded paths).
    We do NOT apply np.argmax — predictions are already the label IDs.
    """
    predictions, labels = p

    true_predictions = [
        [ID2LABEL[int(pred)] for (pred, label) in zip(prediction, label_seq) if label != -100]
        for prediction, label_seq in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[int(label)] for (pred, label) in zip(prediction, label_seq) if label != -100]
        for prediction, label_seq in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def train_crf_teacher():
    TEACHER_CRF_DIR = "./teacher_model_crf"

    print("Loading dataset...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)

    print(f"Initializing Custom BERT-CRF Model from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = BanglaBertCRF(MODEL_NAME, num_labels=len(LABEL2ID))

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir="./results_teacher_crf",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,       # CRF converges slower, needs more epochs
        warmup_ratio=0.1,          # LR warmup prevents early instability
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs_teacher_crf",
        logging_steps=100,
        fp16=torch.cuda.is_available(),   # Use mixed precision on GPU
        dataloader_pin_memory=True,
        # CRITICAL: Our model handles collation via custom forward(), not HuggingFace's default
        remove_unused_columns=False,
        save_only_model=True,             # Avoid saving massive optimizer states (1.2GB each)
        save_total_limit=1,               # Only keep the single best checkpoint
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting BanglaBERT-CRF Teacher Training...")
    trainer.train()

    print("\nEvaluating on Test Set...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"CRF Teacher Test Results: {test_results}")

    # Save model: custom nn.Module needs torch.save, not save_pretrained
    os.makedirs(TEACHER_CRF_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(TEACHER_CRF_DIR, "pytorch_model.bin"))
    tokenizer.save_pretrained(TEACHER_CRF_DIR)

    # Save config so we can reload the model architecture
    import json
    model_config = {
        "model_name": MODEL_NAME,
        "num_labels": len(LABEL2ID),
        "label2id": LABEL2ID,
        "id2label": {str(k): v for k, v in ID2LABEL.items()}
    }
    with open(os.path.join(TEACHER_CRF_DIR, "crf_config.json"), "w") as f:
        json.dump(model_config, f, indent=2, ensure_ascii=False)

    print(f"\nCRF Teacher model saved to {TEACHER_CRF_DIR}")


if __name__ == "__main__":
    train_crf_teacher()
