"""
    @Author: Minh Duc
    @Since: 12/25/2021 1:44 PM
"""
import datetime

import pandas as pd

from Trainning.dataset import Dataset
from Trainning.recommenders.utils.constants import SEED as DEFAULT_SEED
from Trainning.recommenders.utils.constants import *
from Trainning.ncf_singlenode import NCF

DEFAULT_USER_COLUMNS = ["userID", "age", "gender", "occupation", "zip-code"]
DEFAULT_MOVIE_COLUMNS = ['itemID', 'movie_title', 'release_date', 'date',
                     'url', 'unknown', 'Action', 'Adventure', 'Animation',
                     'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                     'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                     'Thriller', 'War', 'Western']


class NCFModel:
    def __init__(self):
        self.train_data = pd.read_csv('./data/ml-100k/u.data', delimiter='\t', names=DEFAULT_HEADER)
        self.train_data['timestamp'] = self.train_data['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))

        self.data = Dataset(train=self.train_data, seed=DEFAULT_SEED)
        self.item_data = pd.read_csv('data/ml-100k/u.item', encoding='cp1252', delimiter='|', names=DEFAULT_MOVIE_COLUMNS)
        self.user_data = pd.read_csv('data/ml-100k/u.user', delimiter='|', names=DEFAULT_USER_COLUMNS)

        movie_columns = DEFAULT_MOVIE_COLUMNS
        self.genre_columns = sorted(movie_columns[5:])
        self.item_data = self.item_data.drop(columns=['date'])
        self.item_data = self.item_data.assign(
            genre=self.item_data.filter(self.genre_columns).pipe(lambda d: d.columns[d.values.argmax(1)]))
        self.item_data = self.item_data.drop(columns=self.genre_columns)
        self.item_data = self.item_data.drop(columns=['url'])

        self.train_data = self.train_data.join(self.user_data.set_index('userID'), on='userID')
        self.train_data = self.train_data.join(self.item_data.set_index('itemID'), on='itemID')

        self.model = NCF(
            n_users=self.data.n_users,
            n_items=self.data.n_items,
            model_type="NeuMF",
            seed=DEFAULT_SEED,
        )
        self.model.setData(self.data)
        self.model.load(neumf_dir='Trainning/model_checkpoint_neumf')

    def predict(self, userID, itemID):
        return self.model.predict(userID, itemID)

    def get_recommendations(self, userID, limit):
        itemIDList = self.item_data['itemID'].unique()
        predictions = {}
        for itemID in itemIDList:
            predictions[itemID] = self.predict(userID, itemID)

        predictions = dict(sorted(predictions.items(), key=lambda x: x[1]))
        return reversed(list(predictions.items())[-limit:])


