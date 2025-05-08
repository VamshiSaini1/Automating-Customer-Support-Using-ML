"""
eda_visualizations.py

This script performs exploratory data analysis (EDA) on the cleaned dataset by:
1. Visualizing the distribution of sentiments (Positive, Neutral, Negative).
2. Generating word clouds for positive and negative reviews.

Usage:
- Ensure 'cleaned_dataset.csv' exists in the same directory (generated from clean_data.py).
- Run this script to generate visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Load the cleaned dataset
file_path = 'cleaned_dataset.csv'  # Ensure this file exists
data = pd.read_csv(file_path)

# Visualize sentiment distribution
def plot_sentiment_distribution(data):
    sentiment_counts = data['sentiment'].value_counts()
    plt.figure(figsize=(8, 5))
    sentiment_counts.plot(kind='bar', color=['green', 'blue', 'red'])
    plt.title('Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

# Generate word clouds for positive and negative reviews
def generate_word_cloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()

def word_clouds(data):
    positive_reviews = ' '.join(data[data['sentiment'] == 'Positive']['cleaned_review'])
    negative_reviews = ' '.join(data[data['sentiment'] == 'Negative']['cleaned_review'])

    generate_word_cloud(positive_reviews, "Word Cloud for Positive Reviews")
    generate_word_cloud(negative_reviews, "Word Cloud for Negative Reviews")

# Execute EDA
if __name__ == "__main__":
    print("Visualizing sentiment distribution...")
    plot_sentiment_distribution(data)
    print("Generating word clouds for positive and negative reviews...")
    word_clouds(data)
