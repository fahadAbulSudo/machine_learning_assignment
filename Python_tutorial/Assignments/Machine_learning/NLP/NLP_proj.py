import streamlit as st
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from io import StringIO
import pickle
import re

from tensorflow.keras.models import load_model
savedModel=load_model('gfgModel.h5')
savedModel.summary()

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, reviews):
        return [self.wnl.lemmatize(t) for t in word_tokenize(reviews)]

pickledCV_model = pickle.load(open('CV_model.pkl', 'rb'))
pickled_model = pickle.load(open('MNB_model.pkl', 'rb'))

words = dict()
tokenizer = nltk.RegexpTokenizer(r"\w+")
lemmatizer = WordNetLemmatizer()


def add_to_dict(d, filename):
  with open(filename, 'r', encoding="utf8") as f:
    for line in f.readlines():
      line = line.split(' ')

      try:
        d[line[0]] = np.array(line[1:], dtype=float)
      except:
        continue
add_to_dict(words, 'glove.6B.50d.txt')


def message_to_token_list(s):
  tokens = tokenizer.tokenize(s)
  lemmatized_tokens = [lemmatizer.lemmatize(t) for t in tokens]
  useful_tokens = [t for t in lemmatized_tokens if t in words]
  return useful_tokens


def NLP_lstm_pipeline(review, word_dict=words, desired_sequence_length=400):
  review = review.lower()
  email_urls = re.compile("(\bhttp.+? | \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)")
  review = re.sub(email_urls, '', review)
  review = re.sub(r'[^\w\s]', '', review)
  # Remove digits
  pattern = r'[0-9]'
  review = re.sub(pattern, '', review)
  processed_list_of_tokens = message_to_token_list(review)

  vectors = []    
  for token in processed_list_of_tokens:
    if token not in word_dict:
      continue
    
    token_vector = word_dict[token]
    vectors.append(token_vector)

  print(len(vectors))
  sequence_length_difference = desired_sequence_length - len(vectors)
  
  pad = np.zeros(shape=(sequence_length_difference, 50))
  
  vectors = np.array(vectors).astype(float)
  print(vectors.shape)
  vectors = np.concatenate([vectors, pad])
  print(vectors.shape)
  vectors = np.reshape(vectors, (1, 400, 50))
  print(vectors.shape)
  predictions = (savedModel.predict(vectors) > 0.5).astype(int)

  if predictions == 1:
    return "positive review"

  else:
    return "negative review"

def NLP_Naive_pipeline(review):
  list = []
  email_urls = re.compile("(\bhttp.+? | \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b)")
  review = re.sub(email_urls, '', review)
  ats = re.compile('@')
  review = re.sub(ats, 'a', review)
  review = re.sub(r'[^\w\s]', '', review)
  pattern = r'[0-9]'
  review = re.sub(pattern, '', review)
  review = word_tokenize(review)
  #review = [word for word in review if not word in stopwords.words()]
  review = ' '.join(map(str, review))
  print(review)
  list.append(review)
  print(list)
  list = pickledCV_model.transform(list)
  print(list)
  sentiment = pickled_model.predict(list)
  print(sentiment)
  if sentiment[0] == 1:
      return "positive review"
  else:
      return "negative review" 

st.title('Welcome')
st.subheader("Sentiment Analysis NLP App")
st.subheader("Streamlit Projects")
with st.form(key='nlpForm'):
    review = st.text_area("Enter Text Here")
    submit_button = st.form_submit_button(label='Analyze')

add_selectbox = st.sidebar.selectbox(
    "How would you like to predict the sentence?",
    ("LSTM_Model", "Naive_Bayes")
)
if submit_button and add_selectbox == "LSTM_Model":
  Sentiments = NLP_lstm_pipeline(review)
  st.write('The sentiment of review is', Sentiments)
elif submit_button and add_selectbox == "Naive_Bayes":
  Sentiments = NLP_Naive_pipeline(review)
  st.write('The sentiment of review is', Sentiments)