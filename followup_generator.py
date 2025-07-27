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
model_name = "MBZUAI/LaMini-Flan-T5-783M"
local_model_path = "./models/Followup_Generator"

# Download and save model if not already present
if not os.path.exists(local_model_path):
    print("Loading the generator model. This might take a few minutes...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.save_pretrained(local_model_path)
    model.save_pretrained(local_model_path)
else:
    print("Loading generator model from local path...")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                   
def build_followup_prompt(user_query, expert_category, extracted_attrs, missing_attrs):
    return f"""
You are an AI assistant helping users provide complete information to route their queries to the correct expert.
Example:
Expert Category: Medical Consultant
User Query: "Hello, this is a 42 years, male, from Bhopal. My BP has been very high and I feel dizzy often. Issue began around 10 days ago. I'm taking BP tablets. History of hypertension."
Missing Attributes: ['CONTACT_DETAILS', 'FULL_NAME']

Follow-up Questions:
1. How can we contact you?
2. Could you please provide the full name of the patient?

Now follow the same pattern for the below input.

Expert Category: {expert_category}
User Query: {user_query}
Missing Attributes: {repr(missing_attrs)}

Consider only the Missing_Information list and Generate a single question for each element in the Missing_Information list asking user to provide that missing information. Phrase the questions in a polite and user-friendly tone.

Follow-up Questions:"""

def get_followup_questions_llm(user_query, expert_category, extracted_attrs, missing_attrs):
    prompt = build_followup_prompt(user_query, expert_category, extracted_attrs, missing_attrs)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=128)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



