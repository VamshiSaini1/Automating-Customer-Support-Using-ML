# 🧠 Sentiment Analysis of E-Commerce Product Reviews

A full-stack machine learning application that analyzes customer reviews in real time, classifies them into sentiments, generates appropriate replies, and tracks negative feedback using a ticketing system.

## 📌 Project Objectives

- Classify customer feedback into **positive**, **neutral**, or **negative** sentiments.
- Automatically **generate replies** based on detected sentiment.
- Assign **ticket numbers** to negative feedback for issue tracking and resolution.

---

## 🛠️ Tools & Technologies

- **Language:** Python  
- **ML Libraries:** scikit-learn, NLTK, Hugging Face Transformers (BERT)  
- **Vectorization:** Bag of Words, TF-IDF  
- **Backend:** Flask  
- **Database:** MySQL  
- **Deployment:** Flask API with real-time sentiment classification

---

## 🧪 Methodology

### 1. Data Collection & Preprocessing
- Source: Kaggle Flipkart reviews dataset  
- Sentiment Mapping: Based on ratings (1–5)  
  - 4–5: Positive, 3: Neutral, <3: Negative
- Preprocessing: Lowercasing, regex cleanup, stop word removal, tokenization

### 2. Feature Engineering
- **Bag of Words (BoW)** and **TF-IDF** used for classical ML models  
- Features stored as CSV for easy reuse

### 3. Machine Learning Models
- **Logistic Regression, SVM, Random Forest**: Moderate performance
- **Final Model – Fine-tuned BERT**:
  - Accuracy: 91%
  - Macro F1-score: 0.71
  - Best results for positive and negative sentiment

### 4. Web Framework: Flask
- Routes for:
  - Product listing
  - Comment submission and sentiment classification
  - Reply generation and ticket assignment

### 5. Database: MySQL
- Tables:
  - `Products`: Product details
  - `Comments`: User comments, sentiment, replies, and ticket IDs

---

## ⚙️ Project Architecture

```bash
📦 SentimentAnalysisApp/
├── app.py                  # Flask backend
├── model/
│   ├── bert_model.pkl      # Fine-tuned BERT
│   └── vectorizers/        # BoW and TF-IDF vectorizers
├── static/
│   └── css/
├── templates/
│   └── index.html          # Web interface
├── data/
│   └── preprocessed_reviews.csv
└── requirements.txt
