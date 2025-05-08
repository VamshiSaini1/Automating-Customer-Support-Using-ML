"""
text_vectorization.py

This script transforms cleaned text data into numerical formats for model training.
1. Uses Bag of Words (BoW) and TF-IDF for feature extraction.
2. Saves the transformed datasets for further analysis.

Usage:
- Ensure 'cleaned_dataset.csv' exists in the same directory (generated from clean_data.py).
- Run this script to generate vectorized datasets.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pickle

# Load the cleaned dataset
file_path = 'cleaned_dataset.csv'
data = pd.read_csv(file_path)

# Extract features and labels
reviews = data['cleaned_review']
labels = data['sentiment']

# Bag of Words (BoW) representation
print("Creating Bag of Words representation...")
bow_vectorizer = CountVectorizer(max_features=5000)  # Limit to top 5000 words
X_bow = bow_vectorizer.fit_transform(reviews)

# Save BoW representation
with open('bow_vectorizer.pkl', 'wb') as file:
    pickle.dump(bow_vectorizer, file)
pd.DataFrame(X_bow.toarray()).to_csv('bow_features.csv', index=False)
print("BoW representation saved to 'bow_features.csv'.")

# TF-IDF representation
print("Creating TF-IDF representation...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(reviews)

# Save TF-IDF representation
with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(tfidf_vectorizer, file)
pd.DataFrame(X_tfidf.toarray()).to_csv('tfidf_features.csv', index=False)
print("TF-IDF representation saved to 'tfidf_features.csv'.")

print("Text vectorization completed.")
