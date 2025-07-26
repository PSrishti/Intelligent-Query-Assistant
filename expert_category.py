"""Expert Category classification - Broker, Legal Advisor, Astrologer, Tech Support, Medical Consultant"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding
import torch

# Load data

with open("data/expert_queries.jsonl", "r") as f:
    lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    print(lines[:2])  # print first 2 lines

df = pd.read_json("data/expert_queries.jsonl", lines=True)
label_encoder = LabelEncoder()
df["labels"] = label_encoder.fit_transform(df["label"])

# Convert to HuggingFace Dataset
dataset = Dataset.from_pandas(df[["text", "labels"]])
dataset = dataset.train_test_split(test_size=0.2)

# Tokenize
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

dataset = dataset.map(tokenize, batched=True)

import evaluate
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Training
training_args = TrainingArguments(
    output_dir="./models/expert_classifier",
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
model.save_pretrained("./models/expert_classifier")
tokenizer.save_pretrained("./models/expert_classifier")

# Save label encoder
import pickle
with open("models/expert_classifier/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)


if __name__ == "__main__":
    #Model testing
    model_path = "./models/expert_classifier"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    with open(f"{model_path}/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    def classify_expert(query: str):
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            label = label_encoder.inverse_transform([pred])[0]
            return label
    
    test_queries = [
    "I need help understanding the stock market",
    "What are the planetary positions for this month?",
    "Can I get legal advice for a tenant dispute?",
    "What should I invest in as a beginner?",
    "Please read my birth chart",
    ]

    for query in test_queries:
        print(f"{query} → {classify_expert(query)}")
