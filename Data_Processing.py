import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
import os


# Prepare Feature Dataset
# Output : X_data (np array)

users = pd.read_csv('ml-100k/u.user', names=['age', 'gender', 'occ', 'zip']).drop(columns=['zip'])

X_data = []
tmp = pd.get_dummies(users)
for i in range(len(tmp)):
    arr = tmp.iloc[i].tolist()
    for j in range(len(arr)):
        if arr[j] == True:
            arr[j] = 1
        elif arr[j] == False:
            arr[j] = 0
    X_data.append(arr)

# Export user_details one_hot header (list)
with open('Z__user_details_header', 'wb') as f:
    pickle.dump(tmp.columns.tolist(), f)

ages = []
for i in range(len(X_data)):
    ages.append(X_data[i][0])
ages = np.array(ages)

scaler = MinMaxScaler(feature_range=(0, 1))
ages = scaler.fit_transform(ages.reshape(-1, 1))
ages = ages.tolist()

i = 0
for age in ages:
    X_data[i][0] = age[0]
    i = i + 1

X_data = np.array(X_data)


# Prepare Label Dataset
# Output : y_data (np array)

movies = pd.read_csv('ml-100k/u.item', delimiter=',,', header=None, engine='python').drop(columns=[0, 1, 2, 3, 4, 13])
genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
          'Documentary', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
          'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies.loc[len(movies)] = 0 * len(movies)
movies.to_csv('tmp', header=None, index=False)
movies = pd.read_csv('tmp', names=genres)
os.remove('tmp')

user_rating = []
ratings = pd.read_csv('ml-100k/u.data', delimiter='\t', names=['uid', 'mid', 'rating', 'time'])
for i in ratings.uid.unique():
    user_rating.append(ratings.loc[ratings.uid == i])

users_fav = []
for df in user_rating:
    ids = df.mid.tolist()
    fav = movies.iloc[ids].sum().sort_values(ascending=False).index.values[0]
    users_fav.append(fav)
le = LabelEncoder()
y_data = le.fit_transform(users_fav)

labels = []
for i in set(y_data):
    x = list(y_data).index(i)
    labels.append([i, users_fav[x]])

y_data = np.array(y_data)


# Splitting and Generating Datasets

with open('Z__labels', 'wb') as f:
    pickle.dump(labels, f)
    f.close()

X_data, y_data = shuffle(X_data, y_data)
X_train, X_test = train_test_split(X_data, test_size=0.2)
y_train, y_test = train_test_split(y_data, test_size=0.2)

np.save('Z__X_train', X_train)
np.save('Z__y_train', y_train)
np.save('Z__X_test', X_test)
np.save('Z__y_test', y_test)


# import user ratings dataset for preprocessing
# output : extended movies dataset with ratings analytics and genres

df = pd.read_csv('ml-100k/u.data', delimiter='\t', names=['uid', 'mid', 'rating', 'time']).drop(columns=['time'])
count = df.groupby('mid')['rating'].count().tolist()
mean = df.groupby('mid')['rating'].mean().tolist()

# Score calculating algorithm
# Purpose : to balance average_rating and rating_count
r = 3
w = np.array(count).mean()
score = []
for i in range(len(count)):
    score.append((r*w + count[i]*mean[i]) / (w + count[i]))
mid = list(set(df.mid.unique().tolist()))
tmp = {'mid':mid, 'count':count, 'mean':mean, 'score':score}
df = pd.DataFrame(tmp)

columns = ['mid', 'name', 'a', 'b', 'c', 'unknown', 'Action', 'Adventure', 'Animation', 'Children',
           'Comedy', 'Crime', 'Documentary', 'd', 'Fantasy','Film-Noir', 'Horror', 'Musical',
           'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('ml-100k/u.item', delimiter=',,', names=columns, engine='python').drop(columns=['a', 'b', 'c', 'd'])
movies = movies.merge(df, on='mid')
movies.to_csv('Z__extended_movies_dataset', index=False)