# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:32:42 2019

@author: DingJunyao
"""

import pandas as pd
import numpy as np


# 效用函数：映射值至0-1之间的数，数值越大越好
def uf(a, amin, amax, positive=True):
    if amin == amax:
        return 1
    elif positive:
        return (a - amin) / (amax - amin)
    else:
        return (amax - a) / (amax - amin)


# 对效用函数的封装，用于矩阵
def ufm(a, positive=True):
    return uf(a, a.min(), a.max(), positive)


# 余弦相似性
def cos_similarity(x, y):
    a = float(x.dot(y.T))
    b = (np.linalg.norm(x) * np.linalg.norm(y))
    if b == 0:
        return 0.0
    else:
        return a / b


# 时间效应模型
def time_effect(delta_t, time_effect_lambda):
    return np.e**(-time_effect_lambda * (delta_t / 86400))


# 权重之和调整为1
def weight_alter(weight):
    return weight / weight.sum()


all_genres = [
    'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

ml_ds_path = './dataset/ml-out-reduce'

# 加载数据集
movies = pd.read_csv(ml_ds_path + '/movies.csv')
users = pd.read_csv(ml_ds_path + '/users.csv')
ratings = pd.read_csv(ml_ds_path + '/ratings.csv')
tags = pd.read_csv(ml_ds_path + '/tags.csv')
tag_scores = pd.read_csv(ml_ds_path + '/tag_scores.csv')

# 清除user表中的邮编列
# （因为不同地区的邮编格式不一样，且有类似，在不提供更多信息的情况下无法比较）
users = users.drop(['Zip-code'], axis=1)

# 量化与标准化各项指标
# users中，性别分成F、M，年龄分成7档（1-56，分度值不定），职业分成21档（0-20）
users['Gender'] = users['Gender'].map({'F': 0, 'M': 1})
for i in ('Age', 'Occupation'):
    users[i] = ufm(users[i])

# # ratings表评分标准化
# ratings['Rating'] = ufm(ratings['Rating'])

# 电影之间的相似度，使用余弦相似性计算
# 读取整合分量
ts = pd.DataFrame()
for i in movies['MovieID']:
    if len(tag_scores[tag_scores['MovieID'] == i]) != 0:
        a = pd.DataFrame(
            tag_scores[tag_scores['MovieID'] == i]['Relevance'].rename(
                columns={'Relevance': i}, inplace=True)).T
        a.columns = [k for k in range(1128)]
        a['MovieID'] = i
        ts = pd.concat([ts, a])
ts = pd.merge(movies, ts, on='MovieID', how='left')
ts = ts.set_index('MovieID')
ts = ts.drop(['Title'], axis=1)
# 填充缺失值，使用同类别的电影的均值来填充；
# 如果没有同属性的电影，则使用该电影类别值为0的类别值也为0的电影的均值
# 如果还是没有，则使用所有电影的均值来填充；
for i in ts[pd.isnull(ts[0])].index:
    movies_same_type = ts[(ts[all_genres] == ts.loc[i, all_genres]).T.all()]
    if len(movies_same_type) == 1:
        # 生成列表，列表中的元素为对应电影中属性值为0的列名
        g = [
            h for h in pd.np.where(ts.loc[i, all_genres] == 0, all_genres,
                                   [None for _ in range(18)]) if h
        ]
        movie_same_type = ts[(ts[g] == ts.loc[i, g]).T.all()]
        if len(movies_same_type) == 1:
            movies_same_type = ts
    ts.loc[i, list(range(1128))] = movies_same_type[list(range(1128))].mean()

# 计算余弦相似性
# 这里为全部处理
s_similar = pd.DataFrame(columns=ts.index, index=ts.index, dtype=np.float)
s_similar.index.name = 'MovieID'
for ii, i in enumerate(s_similar.index):
    s_similar.loc[i, i] = 1
    for j in s_similar.index[ii:]:
        s_similar.loc[i, j] = cos_similarity(ts.loc[i], ts.loc[j])
        s_similar.loc[j, i] = s_similar.loc[i, j]

t_now = ratings['Timestamp'].max()
ratings['Delta_t'] = t_now - ratings['Timestamp']

# 初始化矩阵
s_rate = pd.DataFrame(columns=users.UserID,
                      index=movies.MovieID,
                      dtype=np.float)
# w_rate = pd.DataFrame(columns=users.UserID, index=movies.MovieID,
#                       dtype=np.float)
time_effect_rate = pd.DataFrame(columns=users.UserID,
                                index=movies.MovieID,
                                dtype=np.float)
u_buy = pd.DataFrame(columns=movies.MovieID,
                     index=users.UserID,
                     dtype=np.float)
# u_search = pd.DataFrame(columns=movies.MovieID, index=users.UserID,
#                         dtype=np.float)
# u_browse = pd.DataFrame(columns=movies.MovieID, index=users.UserID,
#                         dtype=np.float)

# 计算评价权重
time_effect_lambda = 0.01
# ratings = pd.merge(ratings,
#                    weight_alter(
#                        ratings.groupby('UserID').count()['MovieID'] /
#                        movies['MovieID'].count()),
#                    on='UserID')
# ratings.rename(columns={
#     'MovieID_x': 'MovieID',
#     'MovieID_y': 'Count'
# },
#                inplace=True)
ratings['TimeEffect'] = time_effect(ratings['Delta_t'], time_effect_lambda)
# for i in movies['MovieID']:
#     ratings.loc[ratings['MovieID'] == i, 'Weight'] = weight_alter(
#         ratings['TimeEffect'] *ratings.loc[ratings['MovieID'] == i, 'Count'])

# 评价、时间效应、购买矩阵
for i in users.UserID:
    s_rate[i] = ratings[ratings.UserID == i][['MovieID',
                                              'Rating']].set_index('MovieID')
    # w_rate[i]= ratings[ratings.UserID == i][['MovieID', 'Weight']].set_index(
    #        'MovieID')
    time_effect_rate[i] = ratings[ratings.UserID == i][[
        'MovieID', 'TimeEffect'
    ]].set_index('MovieID')
    # u_buy[i] = ratings[ratings.UserID == i][['MovieID', 'TimeEffect'
    #                                          ]].set_index('MovieID')
# 使用量矩阵
u_usage = s_rate.mask(s_rate >= 0, 0.1).T

# 用户相似性矩阵
users = users.set_index('UserID')
u_similar = pd.DataFrame(columns=users.index,
                         index=users.index,
                         dtype=np.float)
u_similar.index.name = 'UserID'
for ii, i in enumerate(u_similar.index):
    u_similar.loc[i, i] = 1
    for j in u_similar.index[ii:]:
        u_similar.loc[i, j] = cos_similarity(users.loc[i], users.loc[j])
        u_similar.loc[j, i] = u_similar.loc[i, j]

# 导出矩阵
s_similar.to_csv('./temp/s_similar.csv')
s_rate.to_csv('./temp/s_rate.csv')
# w_rate.to_csv('./temp/w_rate.csv')
time_effect_rate.to_csv('./temp/time_effect_rate.csv')
# u_buy.to_csv('./temp/u_buy.csv')
u_usage.to_csv('./temp/u_usage.csv')
u_similar.to_csv('./temp/u_similar.csv')
