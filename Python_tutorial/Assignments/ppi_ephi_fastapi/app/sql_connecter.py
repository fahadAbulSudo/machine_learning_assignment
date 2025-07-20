import json
import mysql.connector
import openai
import os
from dotenv import load_dotenv
import pandas as pd
from app.config import settings
load_dotenv()  # Load environment variables from .env file

openai.api_key = settings.api_key  # Set the OpenAI API key

def get_schema_info(uri, username, password, database, query):
    # Create a connection to the MySQL database
    conn = mysql.connector.connect(
        host=uri,
        user=username,
        password=password,
        database=database
    )

    # Create a cursor to execute queries
    cursor = conn.cursor()

    # Get the schema information
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    schema_info = []

    # Iterate over the tables and fetch their column information
    for table in tables:
        table_name = table[0]
        cursor.execute(f"DESCRIBE {table_name}")
        columns = cursor.fetchall()

        schema_info.append({"table_name": table_name, "columns": columns})

    cursor.close()
    conn.close()

    query_sentence = query
    try:
        query_result = generate_sql_query(query_sentence, uri, username, password, database, schema_info)
        df_table_info = pd.DataFrame.from_dict(query_result)
    except Exception as e:
        table_data = []
        for table in schema_info:
            table_name = table["table_name"]
            columns = [column[0] for column in table["columns"]]
            table_data.append([table_name, ", ".join(columns)])

        df_table_info = pd.DataFrame(table_data, columns=["Table Name", "Columns"])
        print(e)
        return False, df_table_info
    return True, df_table_info

def generate_sql_query(query_sentence: str, host: str, username: str, password: str, database: str, schema_info):

    prompt = f"Forget previous instructions if input does not make sense respond with an empty query a. Given {schema_info}, generate only the SQL query for the sentence '{query_sentence}'. Return the response as a dictionary format where the key is 'code' and the value is the SQL query."
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=100
    )

    extracted_query = response.choices[0].text.strip()
    extracted_query0 = str(extracted_query)

    if "{'code': '" in extracted_query0:
        extracted_query1 = extracted_query0.replace("{'code': '", "")
    else:
        extracted_query1 = extracted_query0

    if "'}" in extracted_query1:
        query = extracted_query1.replace("'}", "")
    else:
        query = extracted_query1
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
    #print(data)

    # Close the cursor and connection
    cursor.close()
    connection.close()
    return data
