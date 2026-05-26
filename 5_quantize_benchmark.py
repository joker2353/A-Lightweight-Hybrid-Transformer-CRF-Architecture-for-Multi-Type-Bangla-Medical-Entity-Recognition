# 5_quantize_benchmark.py
import os
import time
import torch
import torch.quantization
from transformers import AutoModelForTokenClassification, AutoTokenizer
from datasets import load_from_disk
from config import TEACHER_OUT_DIR, PROCESSED_DATA_DIR

def get_model_size(model_path):
    # Calculate size of directory in MB
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)

def get_param_count(model):
    return sum(p.numel() for p in model.parameters())

def benchmark_latency(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()
    
    total_time = 0
    total_samples = 0
    
    # Warmup
    print("  Warming up...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            _ = model(**inputs)
            
    print("  Benchmarking...")
    with torch.no_grad():
        start_time = time.time()
        for batch in dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            _ = model(**inputs)
            total_samples += inputs['input_ids'].size(0)
        end_time = time.time()
        
    total_time = end_time - start_time
    avg_latency_ms = (total_time / total_samples) * 1000
    return avg_latency_ms

def quantize_and_benchmark():
    STUDENT_KD_DIR = "./student_model_kd"
    QUANTIZED_DIR = "./student_model_quantized"
    
    # --- Quantization ---
    print("--- 1. Dynamic INT8 Quantization ---")
    if not os.path.exists(STUDENT_KD_DIR):
        print(f"Error: Distilled student model not found at {STUDENT_KD_DIR}.")
        print("Please ensure 4_distill_student.py has been run completely.")
        return
        
    model = AutoModelForTokenClassification.from_pretrained(STUDENT_KD_DIR)
    
    print("Applying dynamic quantization to Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    os.makedirs(QUANTIZED_DIR, exist_ok=True)
    # Save quantized model using standard torch save (HuggingFace save_pretrained might not support quantized models natively without ONNX)
    torch.save(quantized_model.state_dict(), os.path.join(QUANTIZED_DIR, "pytorch_model.bin"))
    model.config.save_pretrained(QUANTIZED_DIR)
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_KD_DIR)
    tokenizer.save_pretrained(QUANTIZED_DIR)
    print(f"Quantized model saved to {QUANTIZED_DIR}")
    
    # --- Benchmarking ---
    print("\n--- 2. CPU Inference Benchmarking ---")
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    test_dataset = dataset["test"]
    
    from torch.utils.data import DataLoader
    from transformers import default_data_collator
    
    dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=default_data_collator)
    
    # Evaluate Teacher
    print("Benchmarking Teacher...")
    teacher_model = AutoModelForTokenClassification.from_pretrained(TEACHER_OUT_DIR)
    teacher_params = get_param_count(teacher_model)
    teacher_size = get_model_size(TEACHER_OUT_DIR)
    teacher_latency = benchmark_latency(teacher_model, dataloader)
    
    # Evaluate Distilled Student
    print("Benchmarking Distilled Student...")
    student_model = AutoModelForTokenClassification.from_pretrained(STUDENT_KD_DIR)
    student_params = get_param_count(student_model)
    student_size = get_model_size(STUDENT_KD_DIR)
    student_latency = benchmark_latency(student_model, dataloader)
    
    # Evaluate Quantized Student
    print("Benchmarking Quantized Student...")
    quant_size = os.path.getsize(os.path.join(QUANTIZED_DIR, "pytorch_model.bin")) / (1024 * 1024)
    # Re-initialize empty model and load quantized state dict
    quant_eval_model = AutoModelForTokenClassification.from_pretrained(STUDENT_KD_DIR)
    quant_eval_model = torch.quantization.quantize_dynamic(quant_eval_model, {torch.nn.Linear}, dtype=torch.qint8)
    quant_eval_model.load_state_dict(torch.load(os.path.join(QUANTIZED_DIR, "pytorch_model.bin")))
    
    quant_params = get_param_count(quant_eval_model) # Param count won't change drastically, but weights are INT8
    quant_latency = benchmark_latency(quant_eval_model, dataloader)
    
    # --- Results ---
    print("\n================ EFFICIENCY RESULTS ================")
    print(f"{'Model':<25} | {'Params (M)':<10} | {'Size (MB)':<10} | {'Latency (ms/sent)':<20}")
    print("-" * 75)
    print(f"{'Teacher (12-layer)':<25} | {teacher_params/1e6:<10.2f} | {teacher_size:<10.2f} | {teacher_latency:<20.2f}")
    print(f"{'Student (4-layer KD)':<25} | {student_params/1e6:<10.2f} | {student_size:<10.2f} | {student_latency:<20.2f}")
    print(f"{'Quantized Student (INT8)':<25} | {quant_params/1e6:<10.2f} | {quant_size:<10.2f} | {quant_latency:<20.2f}")
    print("====================================================")

if __name__ == "__main__":
    quantize_and_benchmark()
