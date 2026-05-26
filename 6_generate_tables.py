# 6_generate_tables.py
import os
import json
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_from_disk
from config import TEACHER_OUT_DIR, PROCESSED_DATA_DIR
from transformers import DataCollatorForTokenClassification
from transformers import Trainer, TrainingArguments
from evaluate import load as load_metric
import numpy as np
from config import LABEL2ID, ID2LABEL

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

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
        "accuracy": results["overall_accuracy"],
        "macro_f1": results["overall_f1"],
    }

def get_performance(model_dir, dataset):
    if not os.path.exists(model_dir):
        return "N/A", "N/A"
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    training_args = TrainingArguments(
        output_dir="./tmp",
        per_device_eval_batch_size=16,
        logging_dir='./tmp_logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    results = trainer.evaluate()
    return round(results.get("eval_accuracy", 0)*100, 2), round(results.get("eval_macro_f1", 0)*100, 2)

def generate_tables():
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    
    print("Gathering Performance Metrics...")
    t_acc, t_f1 = get_performance(TEACHER_OUT_DIR, dataset)
    s_acc, s_f1 = get_performance("./student_model_no_kd", dataset)
    skd_acc, skd_f1 = get_performance("./student_model_kd", dataset)
    
    # Quantized model eval
    q_acc, q_f1 = "N/A", "N/A"
    quant_dir = "./student_model_quantized"
    if os.path.exists(quant_dir):
        # We need to run standard eval loop for quantized model
        try:
            tokenizer = AutoTokenizer.from_pretrained("./student_model_kd")
            model = AutoModelForTokenClassification.from_pretrained("./student_model_kd")
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            model.load_state_dict(torch.load(os.path.join(quant_dir, "pytorch_model.bin")))
            
            trainer = Trainer(
                model=model,
                args=TrainingArguments(output_dir="./tmp", per_device_eval_batch_size=16),
                eval_dataset=dataset["test"],
                data_collator=DataCollatorForTokenClassification(tokenizer),
                compute_metrics=compute_metrics
            )
            res = trainer.evaluate()
            q_acc, q_f1 = round(res.get("eval_accuracy", 0)*100, 2), round(res.get("eval_macro_f1", 0)*100, 2)
        except Exception as e:
            print(f"Error evaluating quantized model: {e}")
            
    print("\n\nTable 1: Performance Comparison")
    print("="*60)
    print(f"{'Model':<25} | {'Accuracy (%)':<15} | {'Macro F1 (%)':<15}")
    print("-" * 60)
    print(f"{'Ensemble BERT (Paper)':<25} | {'89.58':<15} | {'86.55':<15}")
    print(f"{'BanglaBERT Teacher':<25} | {str(t_acc):<15} | {str(t_f1):<15}")
    print(f"{'TinyBanglaBERT (No KD)':<25} | {str(s_acc):<15} | {str(s_f1):<15}")
    print(f"{'TinyBanglaBERT + KD':<25} | {str(skd_acc):<15} | {str(skd_f1):<15}")
    print(f"{'Quantized Student':<25} | {str(q_acc):<15} | {str(q_f1):<15}")
    print("="*60)

if __name__ == "__main__":
    generate_tables()
