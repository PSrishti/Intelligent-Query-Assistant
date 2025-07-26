import spacy
from spacy.training import Example
from spacy.tokens import DocBin
from pathlib import Path
import random
import os
import argparse

from ner_utils import jsonl_to_spacy, auto_generate_entities_from_file

def train_ner_model(category_name: str, base_model: str = "en_core_web_sm", n_iter: int = 30):
    # === Paths ===
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models" / f"{category_name}_ner"
    TRAIN_SPACY_FILE = DATA_DIR / "train" / f"{category_name}_ner_train_data.spacy"
    TRAIN_JSONL = DATA_DIR / f"{category_name}_ner_train_auto.jsonl"
    ANNOTATED_JSONL = DATA_DIR / f"{category_name}_ner_train.jsonl"

    # === Step 1: Convert JSONL to .spacy format if not already converted ===
    if not TRAIN_SPACY_FILE.exists():
        auto_generate_entities_from_file(TRAIN_JSONL , ANNOTATED_JSONL)
        jsonl_to_spacy(ANNOTATED_JSONL, TRAIN_SPACY_FILE)

    # === Step 2: Load base model ===
    try:
        nlp = spacy.load(base_model)
    except OSError:
        print(f"Model '{base_model}' not found. Run `python -m spacy download {base_model}` first.")
        return

    # === Step 3: Add NER pipe if not present ===
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner", last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Load training data
    doc_bin = DocBin().from_disk(TRAIN_SPACY_FILE)
    docs = list(doc_bin.get_docs(nlp.vocab))

    # Add all labels to the NER pipe
    for doc in docs:
        for ent in doc.ents:
            ner.add_label(ent.label_)

    # === Step 4: Start training ===
    optimizer = nlp.resume_training()
    examples = [
        Example.from_dict(nlp.make_doc(doc.text), {
            "entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        })
        for doc in docs
    ]

    for i in range(n_iter):
        random.shuffle(examples)
        losses = {}
        batches = spacy.util.minibatch(examples, size=8)
        for batch in batches:
            nlp.update(batch, drop=0.3, losses=losses)
        print(f"Iteration {i+1} | Losses: {losses}")

    # === Step 5: Save the trained model ===
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(MODEL_DIR)
    print(f"[✔] Trained model for category '{category_name}' saved to: {MODEL_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model for a given expert category")
    parser.add_argument("--category", required=True, help="Expert category (e.g., broker, astrologer)")
    parser.add_argument("--base_model", default="en_core_web_sm", help="spaCy base model (default: en_core_web_sm)")
    parser.add_argument("--n_iter", type=int, default=30, help="Number of training iterations (default: 30)")
    args = parser.parse_args()

    train_ner_model(args.category.lower(), args.base_model, args.n_iter)
    
#!python attribute_extraction_ner_train.py --category medical_consultant
