"""
    @Author: Minh Duc
    @Since: 12/24/2021 3:12 PM
"""
import datetime

import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from constants import DEFAULT_HEADER

st.title('Group 8')
st.title('Recommender System')
st.title('Neural Collaborative Filtering')
st.markdown("""
This projects use Movilens 100K dataset!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
* **Paper Source:** [paperswithcode.com](https://paperswithcode.com/paper/neural-collaborative-filtering)
* **Model reference:** [github.com/microsoft](https://github.com/microsoft/recommenders)
""")

st.sidebar.header('User Input Features')


# selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990,2020))))


@st.cache
def load_train_data():
    train_data = pd.read_csv('../data/ml-100k/u.data', delimiter='\t', names=DEFAULT_HEADER)
    train_data['timestamp'] = train_data['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    train_data.sort_values(by=['userID', 'itemID'], ascending=[True, True])
    user_data = pd.read_csv('../data/ml-100k/u.user', delimiter='|',
                            names=["userID", "age", "gender", "occupation", "zip-code"])

    movie_columns = ['itemID', 'movie_title', 'release_date', 'date',
                     'url', 'unknown', 'Action', 'Adventure', 'Animation',
                     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western']
    genre_columns = movie_columns[5:]
    item_data = pd.read_csv('../data/ml-100k/u.item', encoding='cp1252', delimiter='|', names=movie_columns)
    item_data = item_data.drop(columns=['date'])
    item_data = item_data.assign(genre=item_data.filter(genre_columns).pipe(lambda d: d.columns[d.values.argmax(1)]))
    item_data = item_data.drop(columns=genre_columns)
    item_data = item_data.drop(columns=['url'])
    train_data = train_data.join(user_data.set_index('userID'), on='userID')
    train_data = train_data.join(item_data.set_index('itemID'), on='itemID')
    return train_data


train_data = load_train_data()

genre = sorted(train_data['genre'].unique())

# Sidebar - Genre selection
selected_genre = st.sidebar.multiselect('Movie Genres', genre, genre)

# Sidebar - Rate selection
rating = [i for i in range(1, 6)]
selected_rating = st.sidebar.multiselect('Ratings', rating, rating)

# Filtering data
df_train_data_selected = train_data[(train_data.genre.isin(selected_genre)) & (train_data.rating.isin(selected_rating))]

st.header('Dataset Movielens 100k Ratings')

st.write('Collected from 943 Users - 1682 Movies')
st.write('Data Dimension: ' + str(df_train_data_selected.shape[0]) + ' rows and ' + str(
    df_train_data_selected.shape[1]) + ' columns.')

st.dataframe(df_train_data_selected)


# Download NBA player stats data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df, name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv">Download CSV File</a>'
    return href


st.markdown(filedownload(df_train_data_selected, "Group8 - MovieLens 100k Dataset.csv"), unsafe_allow_html=True)

# Heatmap
# if st.button('Intercorrelation Heatmap'):
#     st.header('Intercorrelation Matrix Heatmap')
#     df_train_data_selected.to_csv('output.csv', index=False)
#     df = pd.read_csv('output.csv')
#
#     corr = df.corr()
#     mask = np.zeros_like(corr)
#     mask[np.triu_indices_from(mask)] = True
#     with sns.axes_style("white"):
#         f, ax = plt.subplots(figsize=(7, 5))
#         ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
#     st.pyplot()