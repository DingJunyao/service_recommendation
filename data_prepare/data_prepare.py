# -*- coding: utf-8 -*-
"""
数据集的整合和标准化

Author: DingJunyao
Date：2019-05-17 18:23
"""

import pandas as pd

ML_1M_PATH = '../dataset/ml-1m'
ML_20M_PATH = '../dataset/ml-20m'
OUT_PATH = '../dataset/ml-out'

# 加载所有数据集
users_title = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
users = pd.read_csv(ML_1M_PATH + '/users.dat', sep='::', header=None,
                    names=users_title, engine='python')
movies_title = ['MovieID', 'Title', 'Genres']
movies = pd.read_csv(ML_1M_PATH + '/movies.dat', sep='::', header=None,
                     names=movies_title, engine='python')
ratings_title = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings = pd.read_csv(ML_1M_PATH + '/ratings.dat', sep='::', header=None,
                      names=ratings_title, engine='python')
tags = pd.read_csv(ML_20M_PATH + '/genome-tags.csv')
tag_scores = pd.read_csv(ML_20M_PATH + '/genome-scores.csv')

# 对tags、tag_scores的列重命名，使其标准化
tags.rename(columns={'tagId': 'TagID', 'tag': 'Tag'}, inplace=True)
tag_scores.rename(columns={'movieId': 'MovieID', 'tagId': 'TagID',
                           'relevance': 'Relevance'}, inplace=True)

# 清除tag_scores中相对于movies的多余的电影标签记录
tag_scores = tag_scores[tag_scores['MovieID'] <= movies['MovieID'].max()]

# 将movies各电影的分类转化为矩阵
all_genres = ('Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy',
              'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
              'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
              'War', 'Western')
for i in all_genres:
    movies[i] = movies['Genres'].str.contains(i)
    movies[i] = movies[i].map({True: 1, False: 0})
movies = movies.drop(['Genres'], axis=1)

# 输出为csv
users.to_csv(OUT_PATH + '/users.csv', index=False)
movies.to_csv(OUT_PATH + '/movies.csv', index=False)
ratings.to_csv(OUT_PATH + '/ratings.csv', index=False)
tags.to_csv(OUT_PATH + '/tags.csv', index=False)
tag_scores.to_csv(OUT_PATH + '/tag_scores.csv', index=False)
