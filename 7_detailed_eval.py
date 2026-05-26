# 7_detailed_eval.py
import os
import json
import torch
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_from_disk
import evaluate
from config import PROCESSED_DATA_DIR, TEACHER_OUT_DIR, LABEL2ID, ID2LABEL

# Load seqeval metric
metric = evaluate.load("seqeval")

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
    
    # Extract class-wise metrics
    class_metrics = {}
    for key, val in results.items():
        if isinstance(val, dict):
            # val has precision, recall, f1, number
            class_metrics[key] = {
                "precision": round(val["precision"] * 100, 2),
                "recall": round(val["recall"] * 100, 2),
                "f1": round(val["f1"] * 100, 2),
                "support": val["number"]
            }
            
    return {
        "accuracy": round(results["overall_accuracy"] * 100, 2),
        "precision": round(results["overall_precision"] * 100, 2),
        "recall": round(results["overall_recall"] * 100, 2),
        "macro_f1": round(results["overall_f1"] * 100, 2),
        "class_metrics": class_metrics
    }

def run_evaluation(model, tokenizer, dataset, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = model.to(device)
    model.eval()
    
    # We do a manual evaluation loop to avoid Hugging Face trainer overhead and easily extract structured class-wise results
    data_collator = DataCollatorForTokenClassification(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=16, 
        collate_fn=data_collator
    )
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"]
            
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()
            
            all_preds.append(logits)
            all_labels.append(labels.numpy())
            
    # Concatenate all predictions and labels
    # Since batches might have different lengths due to padding, we pad predictions and labels to evaluate
    max_len = max(p.shape[1] for p in all_preds)
    padded_preds = []
    padded_labels = []
    
    for pred, label in zip(all_preds, all_labels):
        b_size, seq_len, num_labels = pred.shape
        # Pad pred
        if seq_len < max_len:
            pad_width = ((0, 0), (0, max_len - seq_len), (0, 0))
            pred = np.pad(pred, pad_width, mode='constant', constant_values=-100)
            # Pad label
            label = np.pad(label, ((0, 0), (0, max_len - seq_len)), mode='constant', constant_values=-100)
        padded_preds.append(pred)
        padded_labels.append(label)
        
    flat_preds = np.concatenate(padded_preds, axis=0)
    flat_labels = np.concatenate(padded_labels, axis=0)
    
    return compute_metrics((flat_preds, flat_labels))

def main():
    print("Loading test dataset...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    test_set = dataset["test"]
    
    results = {}
    
    # Models to evaluate
    models_info = {
        "Teacher (12-layer)": TEACHER_OUT_DIR,
        "Student KD (4-layer)": "./student_model_kd",
        "Student No KD (4-layer)": "./student_model_no_kd"
    }
    
    print("\n--- Phase 1: Class-Wise Performance Metrics ---")
    for name, path in models_info.items():
        if os.path.exists(path):
            print(f"Evaluating {name} from {path}...")
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForTokenClassification.from_pretrained(path)
            metrics = run_evaluation(model, tokenizer, test_set)
            results[name] = metrics
            print(f"  Accuracy: {metrics['accuracy']}% | Macro F1: {metrics['macro_f1']}%")
        else:
            print(f"Path for {name} does not exist: {path}")

    # Quantized model evaluation
    quant_dir = "./student_model_quantized"
    if os.path.exists(quant_dir):
        print("Evaluating Quantized Student (INT8)...")
        tokenizer = AutoTokenizer.from_pretrained("./student_model_kd")
        model = AutoModelForTokenClassification.from_pretrained("./student_model_kd")
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.load_state_dict(torch.load(os.path.join(quant_dir, "pytorch_model.bin")))
        # Run on CPU only
        metrics = run_evaluation(model, tokenizer, test_set, device="cpu")
        results["Quantized Student (INT8)"] = metrics
        print(f"  Accuracy: {metrics['accuracy']}% | Macro F1: {metrics['macro_f1']}%")

    print("\n--- Phase 2: Epoch-by-Epoch Verification ---")
    epoch_results = {}
    for name, results_dir in [("Teacher", "results_teacher"), ("Student KD", "results_student_kd"), ("Student No KD", "results_student_no_kd")]:
        epoch_results[name] = []
        if os.path.exists(results_dir):
            checkpoints = [d for d in os.listdir(results_dir) if d.startswith("checkpoint-")]
            # Sort checkpoints by step count
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            
            for epoch_idx, cp in enumerate(checkpoints, 1):
                cp_path = os.path.join(results_dir, cp)
                print(f"Evaluating {name} at Epoch {epoch_idx} ({cp})...")
                tokenizer = AutoTokenizer.from_pretrained(cp_path)
                model = AutoModelForTokenClassification.from_pretrained(cp_path)
                metrics = run_evaluation(model, tokenizer, test_set)
                epoch_results[name].append({
                    "epoch": epoch_idx,
                    "step": int(cp.split("-")[1]),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"]
                })
        else:
            print(f"Directory {results_dir} does not exist.")
            
    results["epochs"] = epoch_results

    print("\n--- Phase 3: Layer-Wise Truncation Performance ---")
    # We evaluate how the performance changes as we use fewer layers.
    layer_results = {}
    
    # 1. Teacher model layer-wise truncation
    if os.path.exists(TEACHER_OUT_DIR):
        print("Evaluating Teacher model layer-by-layer (1 to 12 layers)...")
        tokenizer = AutoTokenizer.from_pretrained(TEACHER_OUT_DIR)
        model = AutoModelForTokenClassification.from_pretrained(TEACHER_OUT_DIR)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        teacher_layers_perf = []
        original_layers = model.bert.encoder.layer
        
        for num_layers in range(1, 13):
            # Truncate layers
            model.bert.encoder.layer = torch.nn.ModuleList([original_layers[i] for i in range(num_layers)])
            model.config.num_hidden_layers = num_layers
            
            metrics = run_evaluation(model, tokenizer, test_set)
            teacher_layers_perf.append({
                "num_layers": num_layers,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"]
            })
            print(f"  Teacher with {num_layers} layers -> Accuracy: {metrics['accuracy']}% | Macro F1: {metrics['macro_f1']}%")
            
        # Restore model
        model.bert.encoder.layer = original_layers
        model.config.num_hidden_layers = 12
        layer_results["Teacher"] = teacher_layers_perf

    # 2. Student KD model layer-wise truncation
    if os.path.exists("./student_model_kd"):
        print("Evaluating Student KD model layer-by-layer (1 to 4 layers)...")
        tokenizer = AutoTokenizer.from_pretrained("./student_model_kd")
        model = AutoModelForTokenClassification.from_pretrained("./student_model_kd")
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        student_layers_perf = []
        original_layers = model.bert.encoder.layer
        
        for num_layers in range(1, 5):
            # Truncate layers
            model.bert.encoder.layer = torch.nn.ModuleList([original_layers[i] for i in range(num_layers)])
            model.config.num_hidden_layers = num_layers
            
            metrics = run_evaluation(model, tokenizer, test_set)
            student_layers_perf.append({
                "num_layers": num_layers,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"]
            })
            print(f"  Student KD with {num_layers} layers -> Accuracy: {metrics['accuracy']}% | Macro F1: {metrics['macro_f1']}%")
            
        # Restore model
        model.bert.encoder.layer = original_layers
        model.config.num_hidden_layers = 4
        layer_results["Student KD"] = student_layers_perf
        
    results["layer_truncation"] = layer_results

    # Save results to a file
    out_file = "detailed_evaluation_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    print(f"\nAll detailed verification results successfully saved to {out_file}!")

if __name__ == "__main__":
    main()
