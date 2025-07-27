import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
from typing import List, Dict
import spacy
from pathlib import Path
import argparse
import os
from transformers import logging
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

logging.set_verbosity_error()

print("Running the Intelligent Query Assistant. Please wait a few seconds...\n")

# Expectation schema for each expert
MANDATORY_ATTRIBUTES = {
    "broker": ["NAME", "LOCATION", "PROPERTY_TYPE", "BUDGET", "CONTACT"],
    "legal_advisor": ["FULL_NAME", "OCCUPATION", "JURISDICTION", "ISSUE_DESCRIPTION", "SUPPORTING_DOCUMENTS"],
    "medical_consultant": ["FULL_NAME", "AGE", "GENDER", "LOCATION", "MEDICAL_CONCERN", "SYMPTOMS_DURATION", "MEDICAL_HISTORY", "CONTACT_DETAILS"],
    "tech_support" : ["name", "device Type", "issue description", "software", "warranty status", "contact details"],
    "astrologer": ["Full Name", "Date of Birth", "Time of Birth", "Place of Birth", "Gender", "Current Concern", "Contact Info"]
}

expert_classifier_model_path = "./models/expert_classifier"
with open(f"{expert_classifier_model_path}/label_encoder.pkl", "rb") as f:
    expert_classifier_label_encoder = pickle.load(f)

def classify_expert(query: str):
    model = AutoModelForSequenceClassification.from_pretrained(expert_classifier_model_path)
    tokenizer = AutoTokenizer.from_pretrained(expert_classifier_model_path)
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        label = expert_classifier_label_encoder.inverse_transform([pred])[0]
        return label

classify_query_model_path = "./models/query_classifier"
with open(f"{classify_query_model_path}/label_encoder.pkl", "rb") as f:
    classify_query_label_encoder = pickle.load(f)

def classify_query(query: str):  
    model = AutoModelForSequenceClassification.from_pretrained(classify_query_model_path)
    tokenizer = AutoTokenizer.from_pretrained(classify_query_model_path)
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        label = classify_query_label_encoder.inverse_transform([pred])[0]
        return label

def extract_attributes(category: str, text: str) -> dict:
    #model_path = Path(__file__).resolve().parent / "models" / f"{category.lower()}_ner"
    safe_category = category.lower().replace(" ", "_")
    model_path = Path.cwd() / "models" / f"{safe_category}_ner"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained NER model not found at {model_path}")
    
    nlp = spacy.load(model_path)
    doc = nlp(text)

    extracted_attributes = {}
    for ent in doc.ents:
        # If the label already exists (e.g., multiple symptoms), you can choose to store them in a list
        if ent.label_ in extracted_attributes:
            # Append multiple values for same label
            if isinstance(extracted_attributes[ent.label_], list):
                extracted_attributes[ent.label_].append(ent.text)
            else:
                extracted_attributes[ent.label_] = [extracted_attributes[ent.label_], ent.text]
        else:
            extracted_attributes[ent.label_] = ent.text

    return extracted_attributes

def find_missing_attributes(category: str, extracted_attrs: Dict[str, str]) -> List[str]:
    required = MANDATORY_ATTRIBUTES.get(category, [])
    return [
        attr for attr in required
        if attr not in extracted_attrs or (
            isinstance(extracted_attrs[attr], str) and not extracted_attrs[attr].strip()
        ) or (
            isinstance(extracted_attrs[attr], list) and not any(i.strip() for i in extracted_attrs[attr])
        )
    ]
    #return [attr for attr in required if attr not in extracted_attrs or not extracted_attrs[attr].strip()]

def process_query(user_query: str) -> Dict:
    result = {
        "Original_Query": user_query,
        "Classification": None,
        "Identified_Category": None,
        "Extracted_Attributes": {},
        "Missing_Info": [],
        "AI_questions_for_missing_info": []
    }

    # Step 1: Classify the query
    classification = classify_query(user_query)
    result["Classification"] = classification

    # Step 2: Determine expert category (if applicable)
    if classification in ["CATEGORY_SEARCH", "NEED_PROBLEM_BASED"]:
        category = classify_expert(user_query)
        result["Identified_Category"] = category
        safe_category = category.lower().replace(" ", "_")
        # Step 3: Extract attributes using NER model for the category
        extracted_attrs = extract_attributes(safe_category, user_query)
        result["Extracted_Attributes"] = extracted_attrs

        # Step 4: Find missing attributes
        missing = find_missing_attributes(safe_category, extracted_attrs)
        result["Missing_Info"] = missing

        # Step 5: Generate follow-up questions using a small LLM 
        from followup_generator import get_followup_questions_llm
        followups = get_followup_questions_llm(user_query, category, extracted_attrs, missing)
        result["AI_questions_for_missing_info"] = followups

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Intelligent Query Assistant")
    parser.add_argument("--text", help="Single user query to process", type=str)
    args = parser.parse_args()

    if args.text:
        output = process_query(args.text)
        print("\n===== Query Result =====\n")
        for key, value in output.items():
            print(f"{key}: {value}\n")
    else:
        print("Re-run with the user query.")
        
    """# Sample test queries
    test_queries = [
        "Hey, I'm Srishti. I need a 2BHK flat in Mumbai. My budget is under 50 lakhs. You can call me at 9876543210.",
        "Hi, I’m Sneha Kulkarni, a software engineer facing harassment at my Bangalore workplace. I need help understanding if I can file a complaint under Karnataka labor law.",
        "Hi there, what’s up?",
        "Looking to host a wedding event in Goa next month with a good budget."
        ]

    for query in test_queries:
        output = process_query(query)
        print("\n===== Query Result =====")
        print(output)
"""
