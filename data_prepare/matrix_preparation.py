"""
矩阵的准备

Author: DingJunyao
Date：2019-05-17 18:23
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time


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


all_genres = [
    'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
]

ML_DS_PATH = '../dataset/ml-out-sample'
OUT_PATH = '../temp-sample'

# 加载数据集
movies = pd.read_csv(ML_DS_PATH + '/movies.csv')
users = pd.read_csv(ML_DS_PATH + '/users.csv')
ratings = pd.read_csv(ML_DS_PATH + '/ratings.csv')
tags = pd.read_csv(ML_DS_PATH + '/tags.csv')
tag_scores = pd.read_csv(ML_DS_PATH + '/tag_scores.csv')

prep_start = time.time()  # 打点计时

# 清除user表中的邮编列
# （因为不同地区的邮编格式不一样，且有类似，在不提供更多信息的情况下无法比较）
users = users.drop(['Zip-code'], axis=1)

# 量化与标准化各项指标
# users中，性别分成F、M，年龄分成7档（1-56，分度值不定），职业分成21档（0-20）
users['Gender'] = users['Gender'].map({'F': 0, 'M': 1})
for i in ('Age', 'Occupation'):
    users[i] = ufm(users[i])

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
s_similar = pd.DataFrame(cosine_similarity(ts), columns=ts.index,
                         index=ts.index, dtype=np.float)

# 初始化矩阵
s_rate = pd.DataFrame(columns=users.UserID, index=movies.MovieID,
                      dtype=np.float)
rate_time = pd.DataFrame(columns=users.UserID, index=movies.MovieID,
                         dtype=np.int)

# 评分、时间矩阵
s_rate = s_rate.fillna(0) + ratings.pivot(index='MovieID', columns='UserID',
                                          values='Rating')
rate_time = rate_time.fillna(0) + ratings.pivot(index='MovieID',
                                                columns='UserID',
                                                values='Timestamp')

# 分开新旧评分
ratings_new = ratings.copy()
ratings_old = ratings.copy()
for u in users.UserID:
    ratings_new[(ratings_new['UserID'] == u) & (ratings_new['Timestamp'] < ratings_new[ratings_new['UserID'] == u]['Timestamp'].quantile(0.8))] = np.nan
    ratings_old[(ratings_old['UserID'] == u) & (ratings_old['Timestamp'] > ratings_old[ratings_old['UserID'] == u]['Timestamp'].quantile(0.8))] = np.nan
ratings_new.dropna(inplace=True)
ratings_old.dropna(inplace=True)
s_rate_new = s_rate.copy()
s_rate_new[(rate_time < rate_time.quantile(0.8))] = np.nan
s_rate_old = s_rate.copy()
s_rate_old[(rate_time > rate_time.quantile(0.8))] = np.nan
rate_time_new = rate_time.copy()
rate_time_new[s_rate_new.isnull()] = np.nan
rate_time_old = rate_time.copy()
rate_time_old[s_rate_old.isnull()] = np.nan

# 用户相似性矩阵
users = users.set_index('UserID')
u_similar = pd.DataFrame(cosine_similarity(users), columns=users.index,
                         index=users.index, dtype=np.float)

prep_end = time.time()  # 打点计时

# 导出矩阵
s_similar.to_csv(OUT_PATH + '/s_similar.csv')
ratings_new.to_csv(ML_DS_PATH + '/ratings_new.csv')
ratings_old.to_csv(ML_DS_PATH + '/ratings_old.csv')
s_rate.to_csv(OUT_PATH + '/s_rate.csv')
s_rate_new.to_csv(OUT_PATH + '/s_rate_new.csv')
s_rate_old.to_csv(OUT_PATH + '/s_rate_old.csv')
rate_time_new.to_csv(OUT_PATH + '/rate_time_new.csv')
rate_time_old.to_csv(OUT_PATH + '/rate_time_old.csv')
rate_time.to_csv(OUT_PATH + '/rate_time.csv')
u_similar.to_csv(OUT_PATH + '/u_similar.csv')

print('Time: %s' % (prep_end - prep_start))  # 打点计时
