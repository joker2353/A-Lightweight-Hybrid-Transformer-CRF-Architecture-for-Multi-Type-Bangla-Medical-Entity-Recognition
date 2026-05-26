# 4_distill_student.py
import os
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
from config import MODEL_NAME, PROCESSED_DATA_DIR, TEACHER_OUT_DIR, LABEL2ID, ID2LABEL

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

class KDTrainer(Trainer):
    def __init__(self, teacher_model, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        # Move teacher model to the same device as the student model
        self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Move teacher model to device if not already
        if self.teacher_model.device != model.device:
            self.teacher_model = self.teacher_model.to(model.device)

        # Forward pass student
        outputs_student = model(**inputs)
        logits_student = outputs_student.logits
        
        # Forward pass teacher
        with torch.no_grad():
            outputs_teacher = self.teacher_model(**inputs)
            logits_teacher = outputs_teacher.logits
            
        labels = inputs.get("labels")
        
        # Compute KD loss only for the valid tokens (where labels != -100)
        active_loss = labels.view(-1) != -100
        
        active_logits_student = logits_student.view(-1, model.config.num_labels)[active_loss]
        active_logits_teacher = logits_teacher.view(-1, self.teacher_model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        
        # 1. Hard Cross Entropy Loss
        loss_fct = nn.CrossEntropyLoss()
        loss_ce = loss_fct(active_logits_student, active_labels)
        
        # 2. Soft KL Divergence Loss
        loss_kld = nn.KLDivLoss(reduction="batchmean")
        # Soften probabilities
        student_probs = F.log_softmax(active_logits_student / self.temperature, dim=-1)
        teacher_probs = F.softmax(active_logits_teacher / self.temperature, dim=-1)
        
        loss_kd = loss_kld(student_probs, teacher_probs) * (self.temperature ** 2)
        
        # Combine Loss
        alpha = 0.5
        loss = (1. - alpha) * loss_ce + alpha * loss_kd
        
        return (loss, outputs_student) if return_outputs else loss

def distill_student():
    STUDENT_INIT_DIR = "./student_model_initialized"
    STUDENT_OUT_DIR = "./student_model_kd"
    
    print("Loading dataset...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    
    print("Loading Teacher Model...")
    teacher_model = AutoModelForTokenClassification.from_pretrained(TEACHER_OUT_DIR)
    
    print("Loading Initialized Student Model...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_INIT_DIR)
    student_model = AutoModelForTokenClassification.from_pretrained(STUDENT_INIT_DIR)
    
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./results_student_kd",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir='./logs_student_kd',
    )
    
    trainer = KDTrainer(
        teacher_model=teacher_model,
        temperature=2.0,
        model=student_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    print("Starting Knowledge Distillation Training...")
    trainer.train()
    

    print("Evaluating Distilled Student on Test Set...")
    test_results = trainer.evaluate(dataset["test"])
    print(f"Distilled Student Test Results: {test_results}")
    
    os.makedirs(STUDENT_OUT_DIR, exist_ok=True)
    trainer.save_model(STUDENT_OUT_DIR)
    tokenizer.save_pretrained(STUDENT_OUT_DIR)
    print(f"Distilled Student model saved to {STUDENT_OUT_DIR}")

if __name__ == "__main__":
    distill_student()
