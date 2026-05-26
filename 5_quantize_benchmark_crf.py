# 5_quantize_benchmark_crf.py
"""
Quantization and benchmarking for the full CRF pipeline models.
Extends the original benchmark to include CRF Teacher and CRF-KD Student.

Efficiency metrics (on CPU, simulating edge deployment):
  - Parameter count
  - Model size on disk (MB)
  - Inference latency (ms/sentence)
"""
import os
import time
import json
import torch
import numpy as np
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification
)
from datasets import load_from_disk
from config import PROCESSED_DATA_DIR, TEACHER_OUT_DIR, LABEL2ID, ID2LABEL
from crf_model import BanglaBertCRF

QUANT_STUDENT_CRF_KD_DIR = "./student_model_crf_kd_quantized"


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_size_mb(path: str) -> float:
    """Sum all model weight files in a directory."""
    total = 0
    for fname in ["pytorch_model.bin", "model.safetensors"]:
        fpath = os.path.join(path, fname)
        if os.path.exists(fpath):
            total += os.path.getsize(fpath)
    return round(total / (1024 ** 2), 2)


def benchmark_latency_cpu(model, tokenizer, dataset, n_samples=100) -> float:
    """Measure average CPU inference latency over n_samples test sentences."""
    model = model.to("cpu")
    model.eval()

    data_collator = DataCollatorForTokenClassification(tokenizer)
    subset = dataset.select(range(min(n_samples, len(dataset))))
    dataloader = torch.utils.data.DataLoader(subset, batch_size=1, collate_fn=data_collator)

    latencies = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            start = time.perf_counter()
            _ = model(**inputs)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)

    return round(float(np.mean(latencies)), 2)


def quantize_and_save_student(student_dir: str, save_dir: str):
    """Apply INT8 dynamic quantization to the CRF-KD student and save."""
    print(f"  Loading student from {student_dir} for quantization...")
    model = AutoModelForTokenClassification.from_pretrained(student_dir)

    print("  Applying INT8 dynamic quantization...")
    model_q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model_q.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    print(f"  Quantized model saved to {save_dir}")
    return model_q


def run_benchmarks():
    print("Loading dataset...")
    dataset = load_from_disk(PROCESSED_DATA_DIR)
    test_set = dataset["test"]

    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Original Teacher (12-layer, no CRF) — baseline reference
    # ─────────────────────────────────────────────────────────────────────────
    if os.path.exists(TEACHER_OUT_DIR):
        print("\n[1/5] Benchmarking Original Teacher (12-layer, no CRF)...")
        tokenizer = AutoTokenizer.from_pretrained(TEACHER_OUT_DIR)
        model = AutoModelForTokenClassification.from_pretrained(TEACHER_OUT_DIR)
        params = count_parameters(model)
        size = get_model_size_mb(TEACHER_OUT_DIR)
        latency = benchmark_latency_cpu(model, tokenizer, test_set)
        results["Teacher (12L, no CRF)"] = {
            "params_M": round(params / 1e6, 2),
            "size_MB": size,
            "cpu_latency_ms": latency,
        }
        print(f"  Params: {params/1e6:.2f}M | Size: {size} MB | Latency: {latency} ms")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. CRF Teacher (12-layer + CRF head)
    # ─────────────────────────────────────────────────────────────────────────
    crf_teacher_dir = "./teacher_model_crf"
    if os.path.exists(crf_teacher_dir):
        print("\n[2/5] Benchmarking CRF Teacher (12-layer + CRF)...")
        cfg_path = os.path.join(crf_teacher_dir, "crf_config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        tokenizer = AutoTokenizer.from_pretrained(crf_teacher_dir)
        model = BanglaBertCRF(cfg["model_name"], num_labels=cfg["num_labels"])
        model.load_state_dict(torch.load(
            os.path.join(crf_teacher_dir, "pytorch_model.bin"), map_location="cpu"
        ))
        params = count_parameters(model)
        size_bytes = os.path.getsize(os.path.join(crf_teacher_dir, "pytorch_model.bin"))
        size = round(size_bytes / (1024 ** 2), 2)
        latency = benchmark_latency_cpu(model, tokenizer, test_set)
        results["CRF Teacher (12L + CRF)"] = {
            "params_M": round(params / 1e6, 2),
            "size_MB": size,
            "cpu_latency_ms": latency,
        }
        print(f"  Params: {params/1e6:.2f}M | Size: {size} MB | Latency: {latency} ms")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Original Student KD (4-layer, no CRF) — from previous pipeline
    # ─────────────────────────────────────────────────────────────────────────
    student_kd_dir = "./student_model_kd"
    if os.path.exists(student_kd_dir):
        print("\n[3/5] Benchmarking Original Student KD (4-layer, no CRF)...")
        tokenizer = AutoTokenizer.from_pretrained(student_kd_dir)
        model = AutoModelForTokenClassification.from_pretrained(student_kd_dir)
        params = count_parameters(model)
        size = get_model_size_mb(student_kd_dir)
        latency = benchmark_latency_cpu(model, tokenizer, test_set)
        results["Student KD (4L, no CRF)"] = {
            "params_M": round(params / 1e6, 2),
            "size_MB": size,
            "cpu_latency_ms": latency,
        }
        print(f"  Params: {params/1e6:.2f}M | Size: {size} MB | Latency: {latency} ms")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. CRF-KD Student (4-layer, distilled from CRF Teacher)
    # ─────────────────────────────────────────────────────────────────────────
    student_crf_kd_dir = "./student_model_crf_kd"
    if os.path.exists(student_crf_kd_dir):
        print("\n[4/5] Benchmarking CRF-KD Student (4-layer)...")
        tokenizer = AutoTokenizer.from_pretrained(student_crf_kd_dir)
        model = AutoModelForTokenClassification.from_pretrained(student_crf_kd_dir)
        params = count_parameters(model)
        size = get_model_size_mb(student_crf_kd_dir)
        latency = benchmark_latency_cpu(model, tokenizer, test_set)
        results["Student CRF-KD (4L)"] = {
            "params_M": round(params / 1e6, 2),
            "size_MB": size,
            "cpu_latency_ms": latency,
        }
        print(f"  Params: {params/1e6:.2f}M | Size: {size} MB | Latency: {latency} ms")

        # ─────────────────────────────────────────────────────────────────────
        # 5. Quantized CRF-KD Student (INT8)
        # ─────────────────────────────────────────────────────────────────────
        print("\n[5/5] Quantizing and benchmarking INT8 CRF-KD Student...")
        model_q = quantize_and_save_student(student_crf_kd_dir, QUANT_STUDENT_CRF_KD_DIR)
        params_q = count_parameters(model_q)
        size_q_bytes = os.path.getsize(os.path.join(QUANT_STUDENT_CRF_KD_DIR, "pytorch_model.bin"))
        size_q = round(size_q_bytes / (1024 ** 2), 2)
        latency_q = benchmark_latency_cpu(model_q, tokenizer, test_set)
        results["Quantized CRF-KD Student (INT8)"] = {
            "params_M": round(params_q / 1e6, 2),
            "size_MB": size_q,
            "cpu_latency_ms": latency_q,
        }
        print(f"  Params: {params_q/1e6:.2f}M | Size: {size_q} MB | Latency: {latency_q} ms")

    # ─────────────────────────────────────────────────────────────────────────
    # Print full comparison table
    # ─────────────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 75)
    print("Table 2 (Extended): Efficiency Comparison — Full CRF Pipeline")
    print("=" * 75)
    print(f"{'Model':<35} | {'Params (M)':>10} | {'Size (MB)':>10} | {'CPU ms/sent':>12}")
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<35} | {r['params_M']:>10} | {r['size_MB']:>10} | {r['cpu_latency_ms']:>12}")
    print("=" * 75)

    # Save raw results
    with open("benchmark_crf_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nBenchmark results saved to benchmark_crf_results.json")


if __name__ == "__main__":
    run_benchmarks()
