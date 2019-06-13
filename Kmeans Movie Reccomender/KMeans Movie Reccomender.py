# K-Means Clustering of Movie Rating for Recommendation
# https://programming.rhysshea.com/K-means_movie_ratings/
# Dataset: MovieLens user rating

import pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np


def get_most_rated_movies(movie_ratings, n_movies):
    # add one row = [nr. reviews movie 1, nr. reviews movie 2, ...]
    movie_ratings = movie_ratings.append(movie_ratings.count(), ignore_index=True)
    # sort the movies by the "count" row (nr. of reviews)
    movie_ratings = movie_ratings.sort_values(len(movie_ratings) - 1, axis=1, ascending=False)
    # drop the "count" row
    movie_ratings.drop(movie_ratings.tail(1).index, inplace=True)
    # keep only first n most rated movies
    most_rated_movies = movie_ratings.iloc[:, :n_movies]
    return most_rated_movies


def get_most_active_users(movie_ratings, n_users):
    # add one column = [nr. reviews by user 1, nr. of reviews by user 2, ...]
    movie_ratings['counts'] = pd.Series(movie_ratings.count(axis=1))
    # sort the users by the "count" row (nr. of reviews)
    rated_movies_users = movie_ratings.sort_values('counts', ascending=False)
    # keep only first n more active users
    most_active_users = rated_movies_users.iloc[:n_users, :]
    # drop the "count" row
    most_active_users = most_active_users.drop(['counts'], axis=1)
    return most_active_users


def sort_by_density(movie_ratings, n_movies, n_users):
    most_rated_movies = get_most_rated_movies(movie_ratings, n_movies)
    most_rated_movies_users = get_most_active_users(most_rated_movies, n_users)
    return most_rated_movies_users


def heatmap(movies_df):
    # draw heat map
    plt.pcolor(movies_df)
    plt.yticks(np.arange(0.5, len(movies_df.index), 1), movies_df.index, fontsize=5)
    plt.xticks(np.arange(0.5, len(movies_df.columns), 1), movies_df.columns, fontsize=5, rotation='vertical')
    plt.tight_layout()
    return plt.show()


# read dataset
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
# drop the timestamp column
ratings.drop(['timestamp'], axis=1, inplace=True)
print('The dataset contains:', len(ratings), 'ratings of', len(movies), 'movies.')

ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')
# get rating matrix
user_movie_ratings = pd.pivot_table(ratings_title, values='rating', index='userId', columns='title')

# print a subset
# subset = user_movie_ratings.iloc[:5, :5]
# print('This is a subset:\n', subset)

# show how sparse the user_movie_ratings matrix is
subset = user_movie_ratings.iloc[:50, :50]
heatmap(subset)

# address sparse problem
most_rated_movies = sort_by_density(user_movie_ratings, 50, 70)
sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies).to_coo())

# show much denser the most_rated_movies matrix is
heatmap(most_rated_movies)

# kmeans clustering
num_cluster = 10
kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(sparse_ratings)
clustered = pd.concat([most_rated_movies.reset_index(), pd.DataFrame({'group': kmeans.labels_})], axis=1)


# get user_id
user_id = int(input('login with your ID:'))
print('Welcome back USER', user_id, 'Here are our top movie suggestions for you:')
# get cluster the user belongs to
label = clustered.iloc[user_id, :].group
# build the cluster
cluster = clustered[clustered.group == label].drop(['index', 'group'], axis=1)
heatmap(cluster)
# get the user's movie rating
user_id_ratings = cluster.loc[user_id, :]
# get which movie the user has not seen
unrated_movies = user_id_ratings[user_id_ratings.isnull()]
# get average rating for those movies from the other users in the same cluster
avg_ratings = pd.concat([unrated_movies, cluster.mean()], axis=1, join='inner').loc[:, 0]
# print top recommended (unwatched) movies
avg_ratings = avg_ratings.sort_values(ascending=False)
# avg_ratings.fillna('Sorry, not enough ratings.',inplace=True)
avg_ratings.dropna(inplace=True)
print(avg_ratings.sort_values(ascending=False).head(20))
