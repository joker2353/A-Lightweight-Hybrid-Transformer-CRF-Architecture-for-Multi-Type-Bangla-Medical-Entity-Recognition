# 4_distill_student_from_crf.py
"""
Knowledge Distillation from BanglaBERT-CRF Teacher → TinyBanglaBERT Student.

The key insight: The CRF teacher does NOT output independent per-token logits.
Instead it outputs globally-optimal Viterbi sequences.

Solution: We tap the teacher's PRE-CRF emission scores (the Linear layer output
before the CRF decoding step). These are standard logits [B, seq_len, num_labels]
and serve as excellent soft targets — they encode the teacher's confidence
distribution over all classes, shaped by the CRF training objective.

Loss = α * KL(student_emissions, teacher_emissions) + (1-α) * CE(student, hard_labels)
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from config import PROCESSED_DATA_DIR, LABEL2ID, ID2LABEL
from crf_model import BanglaBertCRF

metric = evaluate.load("seqeval")

CRF_TEACHER_DIR = "./teacher_model_crf"
STUDENT_INIT_DIR = "./student_model_initialized"
STUDENT_CRF_KD_DIR = "./student_model_crf_kd"


def load_crf_teacher(teacher_dir: str) -> BanglaBertCRF:
    """Load the trained CRF teacher model from disk."""
    config_path = os.path.join(teacher_dir, "crf_config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    model = BanglaBertCRF(cfg["model_name"], num_labels=cfg["num_labels"])
    state_dict = torch.load(
        os.path.join(teacher_dir, "pytorch_model.bin"),
        map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_metrics(p):
    """Standard compute_metrics — student uses argmax (no CRF in student)."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [ID2LABEL[pred] for (pred, label) in zip(prediction, label_seq) if label != -100]
        for prediction, label_seq in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[label] for (pred, label) in zip(prediction, label_seq) if label != -100]
        for prediction, label_seq in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


class CRFKDTrainer(Trainer):
    """
    Custom Trainer that distills knowledge from a CRF Teacher into a
    standard linear-head Student.

    Teacher soft targets are extracted from the teacher's EMISSION SCORES
    (pre-CRF linear layer output), giving us clean logits for KL divergence.
    """

    def __init__(self, crf_teacher: BanglaBertCRF, temperature: float = 2.0, alpha: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.crf_teacher = crf_teacher
        self.temperature = temperature
        self.alpha = alpha
        self.crf_teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Move teacher to student's device if needed
        device = next(model.parameters()).device
        if next(self.crf_teacher.parameters()).device != device:
            self.crf_teacher = self.crf_teacher.to(device)

        # --- Student forward pass ---
        outputs_student = model(**inputs)
        logits_student = outputs_student.logits  # [B, seq_len, num_labels]

        # --- Teacher forward pass: extract PRE-CRF emission scores ---
        with torch.no_grad():
            teacher_emissions = self.crf_teacher.get_emissions(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"]
            )  # [B, seq_len, num_labels] — raw logits before CRF decoding

        labels = inputs.get("labels")

        # --- Mask: only compute loss on non-padding, non-subword tokens ---
        active_loss = labels.view(-1) != -100

        active_logits_student = logits_student.view(-1, model.config.num_labels)[active_loss]
        active_logits_teacher = teacher_emissions.view(-1, model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]

        # 1. Hard Cross-Entropy Loss (student vs ground truth labels)
        loss_ce = nn.CrossEntropyLoss()(active_logits_student, active_labels)

        # 2. Soft KL Divergence Loss (student emissions vs teacher CRF-shaped emissions)
        T = self.temperature
        student_log_probs = F.log_softmax(active_logits_student / T, dim=-1)
        teacher_probs = F.softmax(active_logits_teacher / T, dim=-1)
        loss_kd = nn.KLDivLoss(reduction="batchmean")(student_log_probs, teacher_probs) * (T ** 2)

        # 3. Combined loss
        loss = (1.0 - self.alpha) * loss_ce + self.alpha * loss_kd

        return (loss, outputs_student) if return_outputs else loss


def distill_from_crf():
    print("Loading dataset...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)

    print(f"Loading CRF Teacher from {CRF_TEACHER_DIR}...")
    crf_teacher = load_crf_teacher(CRF_TEACHER_DIR)

    print(f"Loading Student from {STUDENT_INIT_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_INIT_DIR)
    student_model = AutoModelForTokenClassification.from_pretrained(STUDENT_INIT_DIR)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir="./results_student_crf_kd",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir="./logs_student_crf_kd",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        save_only_model=True,             # Avoid saving massive optimizer states (800MB each)
        save_total_limit=1,               # Only keep the single best checkpoint
    )

    trainer = CRFKDTrainer(
        crf_teacher=crf_teacher,
        temperature=2.0,
        alpha=0.5,
        model=student_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting CRF-KD Student Training...")
    trainer.train()

    print("\nEvaluating Distilled Student on Test Set...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"CRF-KD Student Test Results: {test_results}")

    os.makedirs(STUDENT_CRF_KD_DIR, exist_ok=True)
    trainer.save_model(STUDENT_CRF_KD_DIR)
    tokenizer.save_pretrained(STUDENT_CRF_KD_DIR)
    print(f"CRF-KD Student model saved to {STUDENT_CRF_KD_DIR}")


if __name__ == "__main__":
    distill_from_crf()
