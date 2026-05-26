# 1_prepare_data.py
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from config import DATA_CSV, MODEL_NAME, ENTITY_MAP, LABEL2ID, PROCESSED_DATA_DIR

def preprocess_and_align():
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_csv(DATA_CSV).fillna("")
    
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    
    for idx, row in df.iterrows():
        text = str(row['Medical Text']).strip()
        if not text:
            continue
            
       # Step A: Create a character-level label array and collision tracker
        char_labels = ['O'] * len(text)
        is_tagged = [False] * len(text) # Prevents substrings from overwriting
        
        # Step B (UPDATED): Map entities (Longest First)
        all_entities = []
        for col_name, tag_suffix in ENTITY_MAP.items():
            if col_name in row and row[col_name]:
                # Split comma-separated entities
                entities = [e.strip() for e in str(row[col_name]).split(',') if e.strip()]
                for ent in entities:
                    all_entities.append((ent, tag_suffix))
        
        # Sort by length descending (e.g. "চোখের এলার্জি" comes before "চোখ")
        all_entities.sort(key=lambda x: len(x[0]), reverse=True)
        
        for ent, tag_suffix in all_entities:
            start_idx = text.find(ent)
            while start_idx != -1:
                end_idx = start_idx + len(ent)
                
                # Only tag if this specific character range hasn't been tagged by a longer word
                if not any(is_tagged[start_idx:end_idx]):
                    char_labels[start_idx] = f'B-{tag_suffix}'
                    is_tagged[start_idx] = True
                    for i in range(start_idx + 1, end_idx):
                        char_labels[i] = f'I-{tag_suffix}'
                        is_tagged[i] = True
                        
                # Look for next occurrence in the text
                start_idx = text.find(ent, start_idx + 1)

        # Step C: Tokenize text and get character offsets
        tokenized = tokenizer(
            text, 
            truncation=True, 
            max_length=512, 
            return_offsets_mapping=True
        )
        
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        offsets = tokenized['offset_mapping']
        
        # Step D: Align Character tags to Tokens
        token_labels = []
        for i, (start, end) in enumerate(offsets):
            if start == end: 
                # Special tokens like [CLS], [SEP], [PAD] get -100 (ignored by PyTorch Loss)
                token_labels.append(-100)
            else:
                # Assign the label of the first character of the token
                char_label = char_labels[start]
                token_labels.append(LABEL2ID[char_label])
                
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(token_labels)

    # Create HuggingFace Dataset
    hf_dataset = Dataset.from_dict({
        'input_ids': all_input_ids,
        'attention_mask': all_attention_masks,
        'labels': all_labels
    })

    # Train/Val/Test Split (80/10/10)
    train_test_split_dict = hf_dataset.train_test_split(test_size=0.2, seed=42)
    val_test_split = train_test_split_dict['test'].train_test_split(test_size=0.5, seed=42)
    
    final_dataset = DatasetDict({
        'train': train_test_split_dict['train'],
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    })

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    final_dataset.save_to_disk(PROCESSED_DATA_DIR)
    print(f"Data successfully processed and saved to {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    preprocess_and_align()