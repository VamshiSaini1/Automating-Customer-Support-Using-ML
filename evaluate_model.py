"""
evaluate_model.py

This script evaluates the fine-tuned BERT model on the validation/test dataset.
1. Loads the fine-tuned model and tokenizer from './bert_fine_tuned'.
2. Evaluates the model using Hugging Face's Trainer.
3. Prints metrics like accuracy, precision, recall, and F1-score.

Usage:
- Ensure 'cleaned_dataset.csv' exists and './bert_fine_tuned' contains the saved model.
- Install required libraries: pip install transformers datasets torch scikit-learn
"""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Define paths
file_path = 'cleaned_dataset.csv'
model_path = './bert_fine_tuned'

# Step 1: Load the cleaned dataset and preprocess it
print("Loading dataset for evaluation...")
data = pd.read_csv(file_path)

# Map sentiments to numerical labels
label_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
data['label'] = data['sentiment'].map(label_mapping)

# Split into validation/test set (use same split logic as training)
_, val_texts, _, val_labels = train_test_split(
    data['cleaned_review'], data['label'], test_size=0.2, random_state=42
)

# Convert validation set to Hugging Face Dataset
val_data = Dataset.from_dict({'text': val_texts, 'label': val_labels})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Tokenize the validation data
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

val_data = val_data.map(tokenize_function, batched=True)
val_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Step 2: Load the fine-tuned model
print("Loading fine-tuned model...")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Step 3: Use Hugging Face Trainer to evaluate
print("Evaluating model...")
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=val_data,
)

# Get evaluation results
results = trainer.evaluate()

# Print evaluation results
print("Evaluation Results:")
print(results)

# Step 4: Get detailed metrics
print("\nCalculating detailed metrics...")
val_preds = trainer.predict(val_data)
y_true = val_labels.tolist()
y_pred = val_preds.predictions.argmax(-1).tolist()

print(classification_report(y_true, y_pred, target_names=label_mapping.keys()))
