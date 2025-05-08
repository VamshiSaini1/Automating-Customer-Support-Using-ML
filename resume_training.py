"""
resume_training.py

This script resumes training for a fine-tuned BERT model using Hugging Face's Trainer.
1. Loads the dataset from 'cleaned_dataset.csv'.
2. Resumes training from the latest checkpoint.
3. Saves the final fine-tuned model and tokenizer.

Usage:
- Ensure 'cleaned_dataset.csv' exists in the same directory.
- Replace 'checkpoint-<number>' with the actual checkpoint folder name in './bert_fine_tuned'.
- Install required libraries: pip install transformers datasets torch scikit-learn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Define paths
file_path = 'cleaned_dataset.csv'  # Ensure this file exists
checkpoint_path = './bert_fine_tuned/checkpoint-938'  # Replace with the latest checkpoint folder name

# Step 1: Load and preprocess the dataset
print("Loading and preprocessing dataset...")
data = pd.read_csv(file_path)

# Map sentiments to numerical labels
label_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
data['label'] = data['sentiment'].map(label_mapping)

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['cleaned_review'], data['label'], test_size=0.2, random_state=42
)

# Convert to Hugging Face Dataset
train_data = Dataset.from_dict({'text': train_texts, 'label': train_labels})
val_data = Dataset.from_dict({'text': val_texts, 'label': val_labels})

# Load tokenizer from the checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

train_data = train_data.map(tokenize_function, batched=True)
val_data = val_data.map(tokenize_function, batched=True)

train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 2: Load the model and set up Trainer
print("Loading model and preparing training...")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

training_args = TrainingArguments(
    output_dir='./bert_fine_tuned',
    eval_strategy="epoch",  # Updated per warning
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # Resume or modify as needed
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
)

# Step 3: Resume training from checkpoint
print("Resuming training from checkpoint...")
trainer.train(resume_from_checkpoint=True)

# Step 4: Save the fine-tuned model and tokenizer
print("Saving the fine-tuned model and tokenizer...")
trainer.save_model('./bert_fine_tuned')
tokenizer.save_pretrained('./bert_fine_tuned')

print("Training resumed and completed successfully.")
