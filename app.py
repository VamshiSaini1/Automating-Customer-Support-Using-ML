from flask import Flask, render_template, request, jsonify
import mysql.connector
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

# Database connection configuration
db_config = {
    'host': 'localhost',
    'user': 'root',        # Replace with your MySQL username
    'password': 'Application@123',    # Replace with your MySQL password
    'database': 'product_comments'    # Replace with your database name
}

# Load the fine-tuned BERT model and tokenizer
model_path = './bert_fine_tuned'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Helper function to classify sentiment and generate a reply
def classify_and_generate_reply(comment_text):
    inputs = tokenizer(comment_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    sentiment_mapping = {2: 'Positive', 1: 'Neutral', 0: 'Negative'}
    sentiment = sentiment_mapping[predicted_class]

    # Generate reply based on sentiment
    if sentiment == 'Positive':
        reply = "Thank you for your positive feedback!"
    elif sentiment == 'Negative':
        reply = "We're sorry to hear about your experience. We'll address your issue as soon as possible."
    else:
        reply = "Thank you for sharing your thoughts!"

    return sentiment, reply

# Function to connect to the database
def get_db_connection():
    connection = mysql.connector.connect(**db_config)
    return connection

# Route to display the main product list page
@app.route('/')
def index():
    return render_template('index.html')

# Route to display a specific product page
@app.route('/product')
def product():
    product_id = request.args.get('product_id')
    return render_template('product.html', product_id=product_id)

# API endpoint to get all products
@app.route('/api/products', methods=['GET'])
def get_products():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Products")
    products = cursor.fetchall()
    cursor.close()
    connection.close()
    return jsonify(products)

# API endpoint to get comments for a specific product
@app.route('/api/product/<int:product_id>', methods=['GET'])
def get_product_comments(product_id):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM Comments WHERE product_id = %s ORDER BY created_at DESC", (product_id,))
    comments = cursor.fetchall()
    cursor.close()
    connection.close()
    return jsonify(comments)

# API endpoint to add a comment to a specific product
@app.route('/api/product/<int:product_id>/comment', methods=['POST'])
def add_comment(product_id):
    data = request.get_json()
    user_comment = data.get('comment_text')

    if not user_comment:
        return jsonify({'error': 'Comment text is required'}), 400

    # Classify the comment and generate a reply
    sentiment, reply_comment = classify_and_generate_reply(user_comment)

    # Generate a ticket number for negative comments
    ticket_number = None
    if sentiment == 'Negative':
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT MAX(ticket_number) FROM Comments WHERE ticket_number IS NOT NULL")
        last_ticket = cursor.fetchone()[0]
        ticket_number = (last_ticket or 1000) + 1  # Start ticket numbers at 1001 if none exist
        cursor.close()
        connection.close()
        reply_comment += f" Your ticket number is {ticket_number}."

    connection = get_db_connection()
    cursor = connection.cursor()

    # Insert the comment into the database with the generated reply and ticket number
    cursor.execute(
        "INSERT INTO Comments (product_id, user_comment, reply_comment, sentiment, ticket_number) VALUES (%s, %s, %s, %s, %s)",
        (product_id, user_comment, reply_comment, sentiment, ticket_number)
    )
    connection.commit()
    cursor.close()
    connection.close()

    return jsonify({'message': 'Comment added successfully!', 'reply_comment': reply_comment, 'ticket_number': ticket_number}), 201


if __name__ == '__main__':
    app.run(debug=True)
