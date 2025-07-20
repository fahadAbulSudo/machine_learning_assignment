import pickle
import streamlit as st
import os

abs = os.getcwd()
abs = abs.replace("\\app\\classification\\Fatal_Health","") 
 
# loading the trained model
classifier = pickle.load(open(abs + "/output/Classification/Fatal_Health/rf_model.pkl", 'rb'))
# classifier = pickle.load(open("rf_model.pkl", 'rb'))

 
@st.cache()
def prediction():
    pass
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(accelerations, uterine_contractions, prolongued_decelerations,
       abnormal_short_term_variability,
       percentage_of_time_with_abnormal_long_term_variability,
       mean_value_of_long_term_variability, histogram_mode,
       histogram_mean,histogram_median, histogram_variance):   
 
 
    # Making predictions 
    prediction = classifier.predict( 
        [[accelerations, uterine_contractions, prolongued_decelerations,
       abnormal_short_term_variability,
       percentage_of_time_with_abnormal_long_term_variability,
       mean_value_of_long_term_variability, histogram_mode,
       histogram_mean,histogram_median, histogram_variance]])
     
    if prediction == 1.0:
        pred = 'Normal'
    elif prediction == 2.0:
        pred = 'Suspected'
    else:
        pred = 'Pathological'    
    return pred
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Fetal Health ML App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 
    accelerations = st.number_input(label="accelerations",step=1.,format="%.2f")
    uterine_contractions = st.number_input(label="uterine_contractions",step=1.,format="%.2f")
    prolongued_decelerations = st.number_input(label="prolongued_decelerations",step=1.,format="%.2f")
    abnormal_short_term_variability = st.number_input(label="abnormal_short_term_variability",step=1.,format="%.2f")
    percentage_of_time_with_abnormal_long_term_variability = st.number_input(label="percentage_of_time_with_abnormal_long_term_variability",step=1.,format="%.2f")
    mean_value_of_long_term_variability = st.number_input(label="mean_value_of_long_term_variability",step=1.,format="%.2f")
    histogram_mode = st.number_input(label="histogram_mode",step=1.,format="%.2f")
    histogram_mean = st.number_input(label="histogram_mean",step=1.,format="%.2f")
    histogram_median = st.number_input(label="histogram_median",step=1.,format="%.2f")
    histogram_variance = st.number_input(label="histogram_variance",step=1.,format="%.2f")
    
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(accelerations, uterine_contractions, prolongued_decelerations,
       abnormal_short_term_variability,
       percentage_of_time_with_abnormal_long_term_variability,
       mean_value_of_long_term_variability, histogram_mode,
       histogram_mean,histogram_median, histogram_variance) 
        st.success('Patient state is {}'.format(result))
       
     
if __name__=='__main__': 
    main()
