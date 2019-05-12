'''
基于SVD矩阵分解的推荐
'''

import numpy as np
import pandas as pd


s_rate = pd.read_csv('./temp/s_rate.csv')
s_rate = s_rate.set_index('MovieID')
s_rate.rename(columns=int, inplace=True)

s_pearson = s_rate - s_rate.mean()
s_pearson = (s_pearson.T - s_pearson.mean(axis=1)).T

U, Sigma, VT = np.linalg.svd(s_pearson.fillna(0).values)

# Sigmap[:1451].sum() / Sigmap.sum() # 0.8

k = 1452
NewData = U[:, :k] * np.mat(np.eye(k) * Sigma[:k]) * VT[:k, :]
ND_DF = pd.DataFrame(NewData, index=s_rate.index, columns=s_rate.columns)

s_predict = (s_pearson.fillna(0) + ND_DF[s_pearson.isnull()].fillna(0))

s_similar = pd.read_csv('./temp/s_similar.csv')
s_similar = s_similar.set_index('MovieID')
s_similar.rename(columns=int, inplace=True)


def recommend_svd(u, num=10, similar_limit=0.8):
    r_1 = s_predict.loc[s_pearson[u].isnull(), u][
        s_predict[u] > 0].sort_values(ascending=False)
    r_2 = r_1
    r_11 = s_predict.loc[~s_pearson[u].isnull(), u][
        s_predict[u] > 0].sort_values(ascending=False)
    for j in r_11.index:
        s_s_1 = s_similar.loc[s_similar.loc[j] < similar_limit, j]
        r_2 = r_2[r_2.index & s_s_1.index]
    rec_movies = dict(r_1[r_2.index & r_1.index][:num])
    return rec_movies


# RMSE
np.sqrt(((s_pearson.fillna(0) - s_predict)**2).sum().sum() /
        (len(s_pearson.columns) * len(s_pearson.index)))


if __name__ == '__main__':
    print(recommend_svd(4, num=10))
