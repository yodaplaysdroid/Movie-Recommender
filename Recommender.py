import pandas as pd
import numpy as np
import pickle
import keras


class User:

    def __init__(self, age, gender, occ):

        self.age = age
        self.gender = gender
        self.occ = occ

    # Recommendation Generation Function
    def recommend(self):

        with open('Z__user_details_header', 'rb') as f:
            headers = pickle.load(f)
        user = np.zeros(len(headers))

        user[headers.index('age')] = self.age
        user[headers.index(f'occ_{self.occ}')] = 1
        user[headers.index(f'gender_{self.gender}')] = 1
        user = np.array([user])

        with open('Z__labels', 'rb') as f:
            labels = pickle.load(f)
        nn = keras.models.load_model('Z__model.h5')
        prediction = nn.predict(x=[user], batch_size=10, verbose=1)

        genres = pd.DataFrame(columns=['prob', 'genre'])
        for i in range(len(labels)):
            genres.loc[i] = [prediction[0][i], labels[i][1]]

        tmp = []
        for i in range(len(genres)):
            tmp.append(int(genres.iloc[i]['prob'] * 20))
        genres['prob'] = tmp
        genres.sort_values('prob', ascending=False), genres['prob'].sum()

        movies = pd.read_csv('Z__extended_movies_dataset')
        recommendations = []
        for index, row in genres.iterrows():
            recommendations = recommendations + movies.loc[movies[row[1]] == 1].sort_values('score', ascending=False).iloc[0:row[0]].name.tolist()
        return list(set(recommendations))

if __name__ == '__main__':
    print(User(50, 'M', 'librarian').recommend())
