import streamlit as st
import pandas as pd
import numpy as np
import datetime
import warnings
import pickle
import time
import os
warnings.filterwarnings("ignore")

abs = os.getcwd()
abs = abs.replace("\\app\\clustering\\Online_retail","")


monthDays = [31, 28, 31, 30, 31, 30,
             31, 31, 30, 31, 30, 31]
 
 
 
def countLeapYears(d):
 
    years = d[0]
    if (d[1] <= 2):
        years -= 1
    return int(years / 4) - int(years / 100) + int(years / 400)
 
def getDifference(dt1, dt2):
    n1 = dt1[0] * 365 + dt1[2]
    for i in range(0, dt1[1] - 1):
        n1 += monthDays[i]
    n1 += countLeapYears(dt1)
    n2 = dt2[0] * 365 + dt2[2]
    for i in range(0, dt2[1]- 1):
        n2 += monthDays[i]
    n2 += countLeapYears(dt2)
    return (n1 - n2)
 
def rec(date_input):
    current_time = datetime.datetime.now()
    current_time = str(current_time)
    lst = current_time.split(" ")
    lst = lst[0]
    lst = lst.split("-") 
    res = [int(x) for x in lst]
    lst1 = date_input.split("-")
    res1 = [int(x) for x in lst1]
    recency = getDifference(res, res1)
    return recency





st.title('Welcome')
st.subheader("Customer Clustering App")
with st.form(key='cluster'):
    monetory = st.number_input('Enter total amount for the customer bought product: ',format="%.2f")
    frequency = st.number_input('Enter frequency of purchases by customer: ',format="%.2f")
    amount = frequency * monetory
    date_input = st.text_input('Enter date in year-month-date format when customer bought product recently: ')
    
    
    
    

    submit_button = st.form_submit_button(label='Predict')
    if submit_button:
        recency = rec(date_input)
        model = pickle.load(open(abs + "\\output\\Clustering\\online_retail\\kmeans.pkl", 'rb'))
        normalize_model = pickle.load(open(abs + "\\output\\Clustering\\online_retail\\normalize_model.pkl", 'rb'))
        nor = []
        nor.append(amount)
        nor.append(frequency)
        nor.append(recency)
        nor = np.reshape(nor, (-1,3 ))
        nor1 = normalize_model.transform(nor)
        cluster = model.predict(nor1)
        if cluster[0] == 0:
            st.write('The customer is TypeA')
        elif cluster[0] == 1:
            st.write('The customer is TypeB') 
        elif cluster[0] == 2:
            st.write('The customer is TypeC') 