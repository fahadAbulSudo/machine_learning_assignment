import streamlit as st
import pandas as pd
import numpy as np
import warnings
import pickle
import time
import sys
import os
warnings.filterwarnings("ignore")

def predict_ind(base_price,checkout_price,wk,emailer_for_promotion):
    abs = os.getcwd()
    abs = abs.replace("\\app\\timeseries\\food_consump","")
    sys.path.insert(0, abs   + "\\utils")
    from common import ra

    model = pickle.load(open(abs + "\\output\\Time Series\\food_consump\\model.pkl", 'rb'))
    standardise_ind = pickle.load(open(abs + "\\output\\Time Series\\food_consump\\standardise_ind.pkl", 'rb'))
    normalise_ind = pickle.load(open(abs + "\\output\\Time Series\\food_consump\\normalise_ind.pkl", 'rb'))
    normalise_ind_num = pickle.load(open(abs + "\\output\\Time Series\\food_consump\\normalise_ind_num.pkl", 'rb'))

    

    nor = []
    nor.append(checkout_price)
    nor.append(base_price)
    nor = np.reshape(nor, (-1,2 ))
    nor1 = normalise_ind.transform(nor)

    main_list = []
    for i in range(0,wk):
        lst = []
        sca = []
        sca.append((145+i+1))
        sca.append(checkout_price)
        sca = np.reshape(sca, (-1,1 ))
        sca1 = standardise_ind.transform(sca)
        lst.append(sca1[0][0])
        lst.append(nor1[0][0])
        lst.append(nor1[0][1])
        lst.append(emailer_for_promotion)
        main_list.append(lst)
#submit_button = st.form_submit_button(label='Predict')
#if st.button("Predict"):
    new_df = pd.DataFrame(columns=['week', 'checkout_price', 'base_price', 'emailer_for_promotion'], data=main_list)
    pred = model.predict(new_df)
    pred = np.reshape(pred, (-1,wk ))
    pred = normalise_ind_num.inverse_transform(pred)
    pred = np.squeeze(pred)
    predic = []
    for i in range(0,len(pred)):
        predic.append(int(pred[i]))
    return ra(predic)




st.title('Welcome')
st.subheader("Food Consumption App for Indian Beverages")
with st.form(key='Predict'):

    Emailer_for_promotion = ["Yes", "No"]

    week = st.number_input('Enter weeks ahead you want to predict: ', min_value=1, max_value=10, value=5, step=1)
    wk = week
    checkout_price = st.number_input('Enter Checkout Price(max:253, min:50): ',format="%.2f")
    base_price = st.number_input('Enter Base Price(max:253): ',format="%.2f")
    email = st.selectbox(

    "Select if Email promotion is done or not from the dropdown",

    Emailer_for_promotion

    )
    if email == "Yes": 
        emailer_for_promotion = 1
    else: 
        emailer_for_promotion = 0
    submit_button = st.form_submit_button(label='Predict')
    if submit_button:
        pred = predict_ind(base_price,checkout_price,wk,emailer_for_promotion)
        main_lst = []
        for i in range(0,len(pred)):
            lst = []
            lst.append(f"Week {i}")
            lst.append(pred[i])
            main_lst.append(lst)
        new_df = pd.DataFrame(columns=['Week', 'Num_orders'], data=main_lst)
        st.dataframe(new_df)