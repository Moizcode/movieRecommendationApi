from typing import final
import streamlit as st
from numpy import loadtxt
import pickle
import pandas as pd
import requests
import base64
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
count = CountVectorizer()
st.set_page_config(layout="wide")

st.markdown('''
<style>
div.css-18e3th9{background-image: url('https://coolbackgrounds.io/images/backgrounds/index/compute-ea4c57a4.png');
background-size: cover;}
.css-1avcm0n {
    display: none;
}
p, ol, ul, dl {
    font-size: 1.5rem;
    font-weight: 400;
    color: lightsalmon;
}
.css-2ykyy6 {
    display: none;
}
.css-1yjuwjr {
    font-size: 1.3rem;
}
.css-18e3th9 {
    padding-top: 4rem;
}
.st-dj {
    font-size: 1.5rem;
}
.st-af {
    font-size: 1.5rem;
}
.st-bq {
    font-size: 1.5rem;
}

</style>
''', unsafe_allow_html=True,
            )

movie_dict = pickle.load(
    open('movie_name.pkl', 'rb'))
movie_list = pd.DataFrame(movie_dict)

movie_detail_dict = pickle.load(
    open('movie_detail.pkl', 'rb'))
movie_details = pd.DataFrame(movie_detail_dict)
final_rec = pd.DataFrame()


def fetch_poster(movie_id):
    response = requests.get(
        'https://api.themoviedb.org/3/movie/{}?api_key=6d5195f330d79be412d6ad8ee5cc8f20'.format(movie_id))
    data = response.json()
    poster_path = data['poster_path']
    return "https://image.tmdb.org/t/p/original/" + poster_path


def get_title_from_index(index):
    return movie_list[movie_list.index == index]["title"].values[0]


def get_index_from_title(title):
    return movie_list[movie_list.title == title].index.values[0]


def recommend(movie_name, rec_type):
    if len(rec_type) == 0:
        rec_type.append('Overall')
    global final_rec
    movie_idx = get_index_from_title(movie_name)
    for n in rec_type:
        if n != 'Overall' and n != 'Storyline':
            st.write(n + ":- " + movie_details.iloc[movie_idx][n])
    final_rec = movie_details[rec_type[0]]
    for t in rec_type[1:]:
        final_rec += movie_details[t]
    matrix = count.fit_transform(final_rec)
    cosine_sim = cosine_similarity(matrix, matrix)

    similar_movies = list(enumerate(cosine_sim[movie_idx]))
    sorted_similar_movie = sorted(
        similar_movies, key=lambda x: x[1], reverse=True)
    recommended_movie = []
    recommended_movie_poster = []
    for i in sorted_similar_movie[1:16]:
        recommended_movie.append(get_title_from_index(i[0]))
        index = get_index_from_title(get_title_from_index(i[0]))
        recommended_movie_poster.append(
            fetch_poster(movie_list.iloc[index]['id']))
    return recommended_movie, recommended_movie_poster


st.title('Movie Recommendation System')


selected = st.selectbox('Select movie', (movie_list['title']))
with st.expander("Filters"):
    categories = st.multiselect('Recommend By', [
        'Director', 'Cast', 'Production companies', 'Genres', 'Storyline', 'Overall'])


if st.button('Recommend'):
    names, poster = recommend(selected, categories)
    col1, col2, col3, col4, col5, = st.columns(5)
    col6, col7, col8, col9, col10, = st.columns(5)
    col11, col12, col13, col14, col15 = st.columns(5)

    with col1:
        st.image(poster[0], use_column_width=True)

    with col2:
        st.image(poster[1], use_column_width=True)

    with col3:
        st.image(poster[2], use_column_width=True)

    with col4:
        st.image(poster[3], use_column_width=True)

    with col5:
        st.image(poster[4], use_column_width=True)

    with col1:
        st.image(poster[5], use_column_width=True)
    with col2:
        st.image(poster[6], use_column_width=True)

    with col3:
        st.image(poster[7], use_column_width=True)

    with col4:
        st.image(poster[8], use_column_width=True)

    with col5:
        st.image(poster[9], use_column_width=True)

    with col1:
        st.image(poster[10], use_column_width=True)
    with col2:
        st.image(poster[11], use_column_width=True)

    with col3:
        st.image(poster[12], use_column_width=True)

    with col4:
        st.image(poster[13], use_column_width=True)

    with col5:
        st.image(poster[14], use_column_width=True)
