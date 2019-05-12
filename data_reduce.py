'''
减少数据，供实验使用
DIV_RAT值为减小数据到几分之一
'''

import pandas as pd
import numpy as np
import os
os.chdir('D:/service_recommendation')

DIV_RAT = 10

# 加载数据集
movies = pd.read_csv('./dataset/ml-out/movies.csv')
users = pd.read_csv('./dataset/ml-out/users.csv')
ratings = pd.read_csv('./dataset/ml-out/ratings.csv')
tag_scores = pd.read_csv('./dataset/ml-out/tag_scores.csv')

max_movieid = list(movies['MovieID'])[int(len(movies['MovieID']) / DIV_RAT) -
                                      1]
max_userid = list(users['UserID'])[int(len(users['UserID']) / DIV_RAT) - 1]

movies = movies[movies['MovieID'] <= max_movieid]
users = users[users['UserID'] <= max_userid]
ratings = ratings[(ratings['MovieID'] <= max_movieid) &
                  (ratings['UserID'] <= max_userid)]
tag_scores = tag_scores[tag_scores['MovieID'] <= max_movieid]

movies.to_csv('./dataset/ml-out-reduce/movies.csv', index=False)
users.to_csv('./dataset/ml-out-reduce/users.csv', index=False)
ratings.to_csv('./dataset/ml-out-reduce/ratings.csv', index=False)
tag_scores.to_csv('./dataset/ml-out-reduce/tag_scores.csv', index=False)
