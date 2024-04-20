
import streamlit as st
import numpy as np
from scipy.sparse.linalg import svds
import pandas as pd
import pickle
import requests
import os

path = 'C:\\Users\\mkafl\\OneDrive\\Desktop\\Movie Recommender'
os.chmod(path,0o755)

def fetch_poster(movie_id):
     url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(movie_id)
     data=requests.get(url)
     data=data.json()
     poster_path = data['poster_path']
     full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
     return full_path

movies = pickle.load(open(".\\movies_list.pkl", 'rb'))
similarity = pickle.load(open(".\\similarity.pkl", 'rb'))
movies_list=movies['title'].values

st.header("Movie Recommender System")



selectvalue=st.selectbox("Select movie from dropdown", movies_list)

def recommend1(movie):
    index=movies[movies['title']==movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    recommend_movie=[]
    recommend_poster=[]
    for i in distance[1:6]:
        movies_id=movies.iloc[i[0]].id
        recommend_movie.append(movies.iloc[i[0]].title)
        recommend_poster.append(fetch_poster(movies_id))
    return recommend_movie, recommend_poster



if st.button("Show Recommend"):
    movie_name, movie_poster = recommend1(selectvalue)
    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
        st.text(movie_name[0])
        st.image(movie_poster[0])
    with col2:
        st.text(movie_name[1])
        st.image(movie_poster[1])
    with col3:
        st.text(movie_name[2])
        st.image(movie_poster[2])
    with col4:
        st.text(movie_name[3])
        st.image(movie_poster[3])
    with col5:
        st.text(movie_name[4])
        st.image(movie_poster[4])




def load_data():
    df1 = pd.read_csv('movies.csv')
    df3 = pd.read_csv('ratings.csv')
    return df1, df3

df1, df3 = load_data()
df = pd.merge(df3, df1, on='movieId')
mean_rating = pd.DataFrame(df.groupby('title')['rating'].mean())
mean_rating['number of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

df6 = df3.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)

df7 = df6.to_numpy()
ratings_mean = np.mean(df7, axis = 1)
mtrx = df7 - ratings_mean.reshape(-1, 1)

a, b, c = svds(mtrx, k = 50)   
b = np.diag(b)

all_predicted_ratings = np.dot(np.dot(a, b), c) + ratings_mean.reshape(-1, 1)  
preds_df = pd.DataFrame(all_predicted_ratings, columns = df6.columns)    

def recommend(preds_df, userId, movie, ratings_df, num_recommendations=5):    
    user_row_number = userId-1 
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) 
    user_data = ratings_df[ratings_df.userId == (userId)]
    user_rated = (user_data.merge(movie, how = 'left', left_on = 'movieId', right_on = 'movieId').
                  sort_values(['rating'], ascending=False)  )
    recommendations = (movie[~movie['movieId'].isin(user_rated['movieId'])].merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',right_on = 'movieId').
               rename(columns = {user_row_number: 'Predictions'}).
               sort_values('Predictions', ascending = False).
               iloc[:num_recommendations, :-1]  )
    return user_rated, recommendations



st.header('Movie recommendations for a given set of users based on collaborative filtering')

st.sidebar.subheader('User Selection')
user_id = st.sidebar.selectbox('Select a user ID', df3['userId'].unique())
num_rec = st.sidebar.slider('Select the number of recommendations', 1, 10, 5)

already_rated, predictions = recommend(preds_df, user_id, df1, df3, num_rec)


st.subheader(f'Recommendations for user {user_id}')
st.table(predictions)

st.subheader(f'Movies that user {user_id} has already rated')
st.table(already_rated.head(5))



