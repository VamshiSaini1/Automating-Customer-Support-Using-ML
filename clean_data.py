"""
clean_data.py

This script cleans a dataset by:
1. Mapping numerical ratings to sentiment categories (Positive, Neutral, Negative).
2. Cleaning the review text by converting to lowercase, and removing special characters and numbers.
3. Saving the cleaned dataset to a new CSV file for further processing.

Usage:
- Replace 'path_to_your_dataset.csv' with the path to your dataset file.
- Run this script to generate 'cleaned_dataset.csv' in the same directory.
"""


import pandas as pd
import re

# Load your dataset
file_path = 'flipkart_reviews_dataset.csv' 
data = pd.read_csv(file_path)

# Focus on relevant columns
cleaned_data = data[['rating', 'review']].copy()

# Map ratings to sentiments
def map_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

cleaned_data['sentiment'] = cleaned_data['rating'].apply(map_sentiment)

# Clean the text: lowercase, remove special characters, numbers, etc.
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and numbers
    return text

cleaned_data['cleaned_review'] = cleaned_data['review'].apply(clean_text)

# Save the cleaned data to a new CSV file
cleaned_data.to_csv('cleaned_dataset.csv', index=False)

print("Data cleaning completed. Cleaned dataset saved as 'cleaned_dataset.csv'.")
