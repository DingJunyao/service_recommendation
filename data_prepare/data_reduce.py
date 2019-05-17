"""
减少数据，供实验使用

DIV_RAT_MOVIE值为抽取电影数据的比例
DIV_RAT_USER值为抽取用户数据的比例
"""

import pandas as pd
import shutil

DIV_RAT_MOVIE = 0.1
DIV_RAT_USER = 0.2

# 加载数据集
DS_IN_PATH = './dataset/ml-out'
DS_OUT_PATH = './dataset/ml-out-sample'

movies = pd.read_csv(DS_IN_PATH + '/movies.csv')
users = pd.read_csv(DS_IN_PATH + '/users.csv')
ratings = pd.read_csv(DS_IN_PATH + '/ratings.csv')
tag_scores = pd.read_csv(DS_IN_PATH + '/tag_scores.csv')

movies_sample = movies.sample(frac=DIV_RAT_MOVIE)
users_sample = users.sample(frac=DIV_RAT_USER)

ratings_sample = ratings[(ratings['MovieID'].isin(movies_sample['MovieID'])) & (
    ratings['UserID'].isin(users_sample['UserID']))]
tag_scores_sample = tag_scores[tag_scores['MovieID'].isin(movies['MovieID'])]

movies_sample.to_csv(DS_OUT_PATH + '/movies.csv', index=False)
users_sample.to_csv(DS_OUT_PATH + '/users.csv', index=False)
ratings_sample.to_csv(DS_OUT_PATH + '/ratings.csv', index=False)
tag_scores_sample.to_csv(DS_OUT_PATH + '/tag_scores.csv', index=False)

shutil.copy(DS_IN_PATH + '/tags.csv', DS_OUT_PATH)
