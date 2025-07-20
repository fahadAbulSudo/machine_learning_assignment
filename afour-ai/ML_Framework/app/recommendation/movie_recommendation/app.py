import streamlit as st
import pickle
import pandas as pd
import requests

from pathlib import Path
import os
import sys


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        # recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names,recommended_movie_posters

st.header('Movie Recommender System')
abs = os.getcwd()
abs = abs.replace("\\app\\recommendation\\movie_recommendation","") 
movie_dict = pickle.load(open(abs + "\\output\\Recommendation\\movie_recommendation\\movie_dict.pkl",'rb'))
similarity = pickle.load(open(abs + "\\output\\Recommendation\\movie_recommendation\\similarity.pkl",'rb'))
movies = pd.DataFrame(movie_dict)

selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movies['title'].values
)


if st.button('Show Recommendation'):
    recommended_movie_names,recommended_movie_posters = recommend(selected_movie)
    col1,col2 = st.columns(2)
    with col1:
        st.text(recommended_movie_names[0])
        st.text(recommended_movie_names[1])
        st.text(recommended_movie_names[2])
        st.text(recommended_movie_names[3])
        st.text(recommended_movie_names[4])
        # st.image(recommended_movie_posters[0])
    with col2:
        pass
        # st.text(recommended_movie_names[1])
    #     # st.image(recommended_movie_posters[1])

    # with col3:
    #     st.text(recommended_movie_names[2])
    #     # st.image(recommended_movie_posters[2])
    # with col4:
    #     st.text(recommended_movie_names[3])
    #     # st.image(recommended_movie_posters[3])
    # with col5:
    #     st.text(recommended_movie_names[4])
    #     # st.image(recommended_movie_posters[4])


