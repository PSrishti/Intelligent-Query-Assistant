""" User query classification
Categories - CATEGORY_SEARCH, NEED_PROBLEM_BASED, INFO_GATHERING_GENERAL, INVALID_NONSENSE """

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch

# Load and encode data

with open("data/query_classification.jsonl", "r") as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print(lines[:2])  # print first 2 lines

df = pd.read_json("data/query_classification.jsonl", lines=True)
label_encoder = LabelEncoder()
df["labels"] = label_encoder.fit_transform(df["label"])

dataset = Dataset.from_pandas(df[["text", "labels"]])
                                                        
# Split
dataset = dataset.train_test_split(test_size=0.2)

# Tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_encoder.classes_)
)

# Training
training_args = TrainingArguments(
    output_dir="./models/query_classifier",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_total_limit=1,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

trainer.train()
model.save_pretrained("./models/query_classifier")
tokenizer.save_pretrained("./models/query_classifier")

# Save label encoder
import pickle
with open("models/query_classifier/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


#from classification.llm_classifier_local import classify_query_local
if __name__ == "__main__":
    #Model testing
    model_path = "./models/query_classifier"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(f"{model_path}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    def classify_query(query: str):
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            label = label_encoder.inverse_transform([pred])[0]
            return label
    
    queries = [
    "Looking for a 2bhk to rent in South Delhi",
    "My laptop screen is flickering",
    "What's the weather like today?",
    "asdfjkl;",
    "Hi!"
    ]

    for query in queries:
        print(f"{query} → {classify_query(query)}")



