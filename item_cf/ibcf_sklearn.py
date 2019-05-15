#!/usr/bin/python3
"""


Author: DingJunyao
Date: 2019-05-13 21:36
"""


import pandas as pd
import numpy as np

header = ['user_id', 'item_id', 'rating', 'timestamp']
dataset = pd.read_csv('../dataset/ml-out/ratings.csv')


#计算唯一用户和电影的数量
# unique对以为数组去重  shape[0] shape为矩阵的长度
users = dataset.UserID.unique().shape[0]
items = dataset.MovieID.unique().shape[0]
from sklearn.model_selection import train_test_split
train_data,test_data = train_test_split(dataset,test_size=0.25)

'''
创建user-item矩阵
itertuples         pandas dataframe 建立索引的方式
结果为：   Pandas(Index=77054, user_id=650, item_id=528, rating=3, timestamp=891370998)
'''
train_data_matrix = np.zeros((users,items))
for line in train_data.itertuples():
    train_data_matrix[line[1] - 1, line[2] - 1] = line[3]

test_data_matrix = np.zeros((users,items))
for line in test_data.itertuples():
    test_data_matrix[line[1] - 1, line[2] - 1] = line[3]
#计算相似度
from sklearn.metrics.pairwise import pairwise_distances
#相似度相当于权重w
user_similarity = pairwise_distances(train_data_matrix,metric='cosine')
#train_data_matrix.T 矩阵转置
items_similarity = pairwise_distances(train_data_matrix.T,metric='cosine')

'''
基于用户相似矩阵 -> 基于用户的推荐
mean函数求取均值  axis=1 对各行求取均值，返回一个m*1的矩阵
np.newaxis 给矩阵增加一个列 一维矩阵变为多维矩阵 mean_user_rating(n*1)
train_data_matrix所有行都减去mean_user_rating对应行的数    此为规范化评分，使其在统一的范围内
numpy a.dot(b) -> 两个矩阵的点积
      np.abs(a) ->计算矩阵a各元素的绝对值
      np.sum()  -> 无参数 矩阵全部元素相加
                -> axis=0   按列相加
                -> axis=1   按行相加
      b /a 矩阵对应为相除
'''
mean_user_rating = train_data_matrix.mean(axis = 1) #计算每行的平均数
rating_diff = train_data_matrix - mean_user_rating[:,np.newaxis]  #评分规范化
pred = mean_user_rating[:, np.newaxis] \
       + user_similarity.dot(rating_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T  #权重w*平均化的评分

'''
评估指标    均方差误差
'''
from sklearn.metrics import mean_squared_error
from math import sqrt

pred = pred[test_data_matrix.nonzero()].flatten()
test_data_matrix = test_data_matrix[test_data_matrix.nonzero()].flatten()
result = sqrt(mean_squared_error(pred,test_data_matrix))
print(result)