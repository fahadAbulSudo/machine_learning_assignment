import streamlit as st
import json
import mysql.connector
import openai
import os
from dotenv import load_dotenv
import pandas as pd
load_dotenv()  # Load environment variables from .env file

openai.api_key = os.getenv("API_KEY")  # Set the OpenAI API key

def get_schema_info(uri, username, password):
    # Create a connection to the MySQL database
    conn = mysql.connector.connect(
        host=uri,
        user=username,
        password=password,
        database=os.getenv("DATABASE")
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

        # Add the table and its columns to the schema information
        schema_info.append({"table_name": table_name, "columns": columns})

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return schema_info


table_info = ""  # Initialize table_info variable

# Retrieve the schema information from the MySQL database
dummy_uri = "localhost"
dummy_username = os.getenv("USER_NAME")
dummy_password = os.getenv("PASSWORD")
table_info = get_schema_info(dummy_uri, dummy_username, dummy_password)
#print(table_info)

# Create the Streamlit application
def main():
    st.title("PII Data Discovery")
    query_sentence = st.text_input("Enter query sentence:")
    if st.button("Fetch Data"):
        if not query_sentence:
            st.error("Query sentence is required.")
            
        else:
            try:
                query_result = generate_sql_query(query_sentence)
                #print(query_result)
                #st.set_page_config(layout="wide")

                df_table_info = pd.DataFrame.from_dict(query_result)
                #
                st.table(df_table_info)
                #st.json(query_result)
                #print(query_result)
            except Exception as e:
                st.error(f"Unable to fetch data please rephrase your sentence")
                st.write("Follow This Table Information:")
                table_data = []
                for table in table_info:
                    table_name = table["table_name"]
                    columns = [column[0] for column in table["columns"]]
                    table_data.append([table_name, ", ".join(columns)])

                df_table_info = pd.DataFrame(table_data, columns=["Table Name", "Columns"])
                st.table(df_table_info)

# Function to generate the SQL query
def generate_sql_query(query_sentence: str):
    if not query_sentence:
        raise ValueError("Query sentence is required.")

    prompt = f"Forget previous instructions if input does not make sense respond with an empty query a. Given {table_info}, generate only the SQL query for the sentence '{query_sentence}'. Return the response as a dictionary format where the key is 'code' and the value is the SQL query."
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
    #print(query)
    # Establish connection to the MySQL database
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
    #print(data)

    # Close the cursor and connection
    cursor.close()
    connection.close()
    return data

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()