"""
model_training.py

This script trains and evaluates machine learning models for sentiment analysis:
1. Uses Logistic Regression, SVM, and Random Forest classifiers.
2. Evaluates models using cross-validation.
3. Saves the best-performing model for deployment.

Usage:
- Ensure 'bow_features.csv' or 'tfidf_features.csv' and their vectorizers exist in the same directory.
- Run this script to compare model performance and save the best model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Load the vectorized data (use either BoW or TF-IDF)
feature_file = 'bow_features.csv'  # Change to 'bow_features.csv' if using BoW
data = pd.read_csv(feature_file)
labels = pd.read_csv('cleaned_dataset.csv')['sentiment']

# Encode labels
label_mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
y = labels.map(label_mapping).values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel='linear'),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Train and evaluate models using cross-validation
best_model = None
best_score = 0
print("Training and evaluating models...")
for model_name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
    avg_score = scores.mean()
    print(f"{model_name} - F1 Score: {avg_score:.4f}")
    if avg_score > best_score:
        best_score = avg_score
        best_model = model_name

# Train the best model on the entire training set
print(f"\nBest model: {best_model}")
final_model = models[best_model]
final_model.fit(X_train, y_train)

# Evaluate on the test set
y_pred = final_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average='macro')
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Save the best model
with open('best_model.pkl', 'wb') as file:
    pickle.dump(final_model, file)
print("Best model saved as 'best_model.pkl'.")
