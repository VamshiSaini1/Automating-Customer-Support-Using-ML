"""
fine_tune_bert.py

This script fine-tunes a pre-trained BERT model for sentiment classification:
1. Loads the cleaned dataset (cleaned_dataset.csv).
2. Fine-tunes BERT using Hugging Face's Transformers.
3. Saves the fine-tuned model for deployment.

Usage:
- Ensure 'cleaned_dataset.csv' exists in the same directory.
- Install required libraries: pip install transformers datasets torch scikit-learn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load the cleaned dataset
file_path = 'cleaned_dataset.csv'
data = pd.read_csv(file_path)

# Map sentiments to numerical labels
label_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
data['label'] = data['sentiment'].map(label_mapping)

# Split the data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['cleaned_review'], data['label'], test_size=0.2, random_state=42
)

# Convert to Hugging Face Dataset
train_data = Dataset.from_dict({'text': train_texts, 'label': train_labels})
val_data = Dataset.from_dict({'text': val_texts, 'label': val_labels})

# Load pre-trained tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)

train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./bert_fine_tuned',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=1,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model('./bert_fine_tuned')
tokenizer.save_pretrained('./bert_fine_tuned')
print("Fine-tuned model and tokenizer saved to './bert_fine_tuned'.")
