# config.py
MODEL_NAME = "sagorsarker/bangla-bert-base"

# CHANGE 1: Use the V2 dataset
DATA_CSV = "MedER_Dataset_Bengali_Entity_V2.csv" 
PROCESSED_DATA_DIR = "./processed_data"
TEACHER_OUT_DIR = "./teacher_model"

# CHANGE 2: Added 'Common Medical Terms' (CMT)
ENTITY_MAP = {
    'Medicine\\ Chemical Name': 'MED',
    'Organ': 'ORG',
    'Disease': 'DIS',
    'Hormone': 'HOR',
    'Pharmacological Class': 'PHA',
    'Common Medical Terms': 'CMT' 
}

# Generate Label Maps
TAGS = ['O']
for tag in ENTITY_MAP.values():
    TAGS.extend([f'B-{tag}', f'I-{tag}'])

LABEL2ID = {tag: i for i, tag in enumerate(TAGS)}
ID2LABEL = {i: tag for tag, i in LABEL2ID.items()}