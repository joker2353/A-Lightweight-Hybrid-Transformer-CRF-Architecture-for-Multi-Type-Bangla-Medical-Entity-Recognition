# 3_create_student.py
import os
import torch
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer
from config import TEACHER_OUT_DIR, MODEL_NAME, LABEL2ID, ID2LABEL

def create_student_model():
    print("Loading Teacher Model...")
    # Load Teacher config
    teacher_config = AutoConfig.from_pretrained(TEACHER_OUT_DIR)
    
    # Create Student Config (4 layers instead of 12)
    student_config = AutoConfig.from_pretrained(TEACHER_OUT_DIR)
    student_config.num_hidden_layers = 4
    
    print("Initializing Student Model (TinyBanglaBERT)...")
    # Initialize Student Model
    student_model = AutoModelForTokenClassification.from_config(student_config)
    
    # Load Teacher Model weights
    teacher_model = AutoModelForTokenClassification.from_pretrained(TEACHER_OUT_DIR)
    
    # Copy Embeddings
    print("Copying Embeddings...")
    student_model.bert.embeddings.load_state_dict(teacher_model.bert.embeddings.state_dict())
    
    # Copy Layers (Teacher: 1, 5, 9, 12 -> 0-indexed: 0, 4, 8, 11)
    # Student: 0, 1, 2, 3
    print("Copying Selected Transformer Layers...")
    layer_mapping = {
        0: 0,
        1: 4,
        2: 8,
        3: 11
    }
    
    for student_layer, teacher_layer in layer_mapping.items():
        print(f"  Mapping Teacher Layer {teacher_layer} -> Student Layer {student_layer}")
        student_model.bert.encoder.layer[student_layer].load_state_dict(
            teacher_model.bert.encoder.layer[teacher_layer].state_dict()
        )
        
    # Copy Classifier Head
    print("Copying Classification Head...")
    student_model.classifier.load_state_dict(teacher_model.classifier.state_dict())
    
    # Note: bert.pooler is not heavily used in Token Classification, but safe to copy
    if getattr(student_model.bert, 'pooler', None) is not None and getattr(teacher_model.bert, 'pooler', None) is not None:
        print("Copying Pooler...")
        student_model.bert.pooler.load_state_dict(teacher_model.bert.pooler.state_dict())

    # Save Student Model
    out_dir = "./student_model_initialized"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Saving Initialized Student Model to {out_dir}...")
    student_model.save_pretrained(out_dir)
    
    # Save Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_OUT_DIR)
    tokenizer.save_pretrained(out_dir)
    print("Done!")

if __name__ == "__main__":
    create_student_model()
