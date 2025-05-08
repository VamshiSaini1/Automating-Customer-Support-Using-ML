import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Application@123",   # Replace with your MySQL password
    database="product_comments"
)