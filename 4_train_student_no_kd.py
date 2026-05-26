# 4_train_student_no_kd.py
import os
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
from config import MODEL_NAME, PROCESSED_DATA_DIR, LABEL2ID, ID2LABEL

# Load NER Evaluation metric
metric = evaluate.load("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (-100)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def train_student_no_kd():
    STUDENT_INIT_DIR = "./student_model_initialized"
    STUDENT_OUT_DIR = "./student_model_no_kd"
    
    print("Loading dataset...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    
    print("Loading Initialized Student Model...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_INIT_DIR)
    student_model = AutoModelForTokenClassification.from_pretrained(STUDENT_INIT_DIR)
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results_student_no_kd",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir='./logs_student_no_kd',
    )
    
    trainer = Trainer(
        model=student_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("Starting Student Training (NO Knowledge Distillation)...")
    trainer.train()
    
    # Evaluate on test set
    print("Evaluating Baseline Student on Test Set...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"Baseline Student Test Results: {test_results}")
    
    os.makedirs(STUDENT_OUT_DIR, exist_ok=True)
    trainer.save_model(STUDENT_OUT_DIR)
    tokenizer.save_pretrained(STUDENT_OUT_DIR)
    print(f"Baseline Student model saved to {STUDENT_OUT_DIR}")

if __name__ == "__main__":
    train_student_no_kd()
