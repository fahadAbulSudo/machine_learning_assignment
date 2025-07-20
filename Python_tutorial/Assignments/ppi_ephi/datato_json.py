import mysql.connector
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv() 

query = "SELECT * FROM Patient;"
host = "localhost"
username = os.getenv("USER_NAME")
password = os.getenv("PASSWORD")
database = os.getenv("DATABASE")
connection = mysql.connector.connect(
    host=host,
    user=username,
    password=password,
    database=database
)

# Execute the SQL query
cursor = connection.cursor(buffered=True , dictionary=True)
cursor.execute(query)
data = cursor.fetchall()
print(data)

# Close the cursor and connection
cursor.close()
connection.close()
    