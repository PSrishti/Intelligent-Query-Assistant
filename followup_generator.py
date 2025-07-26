from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import os
import warnings
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# Disable transformers logging to suppress warnings/info
logging.getLogger("transformers").setLevel(logging.ERROR)

# Lightweight instruction-tuned model
model_name = "google/flan-t5-small"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                   
def build_followup_prompt(user_query, expert_category, extracted_attrs, missing_attrs):
    return f"""
You are an AI assistant helping users provide complete information to route their queries to the correct expert.

Expert Category: {expert_category}
User Query: "{user_query}"
Extracted Attributes: {extracted_attrs}
Missing Attributes: {missing_attrs}

Generate one natural language follow-up question for each missing attribute that is contextually relevant to the query and expert category. Phrase the questions in a polite and user-friendly tone.
"""

def get_followup_questions_llm(user_query, expert_category, extracted_attrs, missing_attrs):
    prompt = build_followup_prompt(user_query, expert_category, extracted_attrs, missing_attrs)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

#prompt = 
"""
You are an AI assistant helping users provide complete information to route their queries to the correct expert.

Expert Category: broker
User Query: "Seeking a rental flat in Powai, Mumbai. Tenant: Surabhi Kulkarni. Monthly budget is 45k. Phone: 9843210987"
Extracted Attributes: {"PROPERTY_TYPE" : "rental flat", "LOCATION" : "Powai, Mumbai", "NAME" : "Surabhi Kulkarni", "CONTACT" : "9843210987"}
Missing Attributes: ['BUDGET', 'ADDITIONAL_FEATURES']

Generate one natural language follow-up question for each missing attribute that is contextually relevant to the query and expert category. Phrase the questions in a polite and user-friendly tone.
"""


