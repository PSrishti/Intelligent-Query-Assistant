import os
import spacy
from spacy.tokens import DocBin
import json
from typing import List, Dict, Tuple

def auto_generate_entities_from_file(input_file: str, output_file: str):
    """
    Reads a JSONL file with `text` and `labels`, auto-generates entity spans,
    and writes a new JSONL file with `text` and `entities` (start, end, label).
    
    input_file (str): Path to the input .jsonl file
    output_file (str): Path to save the transformed .jsonl file
    """
    def extract_entities(text: str, labels: Dict[str, str]) -> Dict[str, List[Tuple[int, int, str]]]:
        entities = []
        for label, val in labels.items():
            if not val:
                continue 
            start = text.find(val)
            if start == -1:
                continue
            end = start + len(val)
            overlap = False
            for s, e, _ in entities:
                if not (end <= s or start >= e):  # overlapping span
                    overlap = True
                    break
            if not overlap:
                entities.append((start, end, label))
        return {"text": text, "entities": entities}

    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    converted_data = [extract_entities(item["text"], item["labels"]) for item in data]

    with open(output_file, "w", encoding="utf-8") as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Conversion complete. Output saved to: {output_file}")


def jsonl_to_spacy(jsonl_path, output_path, nlp=None):
    """
    Converts a JSONL file (with spaCy NER format) to spaCy's binary .spacy format for training.

    Args:
        jsonl_path (str or Path): Path to the .jsonl file.
        output_path (str or Path): Path to save the output .spacy file.
        nlp (Language): Optional spaCy NLP object. If None, will use blank English model.
    """
    valid_examples = 0
    invalid_examples = 0
    if nlp is None:
        nlp = spacy.blank("en")

    db = DocBin()  # spaCy's Doc container
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)
            text = example["text"]
            entities = example.get("entities", [])

            doc = nlp.make_doc(text)
            ents = []
        
            for start, end, label in entities:
                span = doc.char_span(start, end, label=label)
                if span is None:
                    print(f"[WARN] Skipping span: ({start}, {end}, {label}) in text: {text[start:end]}")
                    invalid_examples += 1
                else:
                    ents.append(span)

            if ents:
                doc.ents = ents
                db.add(doc)
                valid_examples += 1
    
    db.to_disk(output_path)
    print(f"[INFO] Saved spaCy training data to: {output_path}")
    print(f"[INFO] Done. Valid examples added: {valid_examples}, Invalid examples skipped: {invalid_examples}")

# Convert training data
#!python -c "from ner_utils import auto_generate_entities_from_file; auto_generate_entities_from_file('data/legal_advisor_ner_train_auto.jsonl', 'data/legal_advisor_ner_train.jsonl')"
#!python -c "from ner_utils import jsonl_to_spacy; jsonl_to_spacy('data/legal_advisor_ner_train.jsonl', 'data/train/legal_advisor_ner_train_data.spacy')"
#!python attribute_extraction_ner.py --category tech_support
#!python main.py --category tech_support --text "Hi, I'm Rajeev and my Samsung Galaxy A52 keeps restarting randomly. It started after a software update. Can you help?"
