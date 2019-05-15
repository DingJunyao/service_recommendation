#!/usr/bin/python3
"""


Author: DingJunyao
Date: 2019-05-13 16:54
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def s_rate_mean(s_rate):
    """
    平均化矩阵内各用户评分

    :param s_rate: 用户评分矩阵：pandas.DataFrame
    :return:       平均化后的用户评分矩阵：pandas.DataFrame
    """
    return s_rate - s_rate.mean()


# def cw_sgd(R, k, alpha=0.005, lambda_v=0.005, epoch=0.0001):
#     U = np.random.rand(R.shape[0], k)
#     V = np.random.rand(R.shape[1], k)
#     S = []
#     for i in range(R.shape[0]):
#         for j in range(R.shape[1]):
#             if R[i][j] == 0:
#                 S.append((i, j))
#     for q in range(k):
#         print('q=%s' % q)
#         convergence = False
#         while not convergence:
#             random.shuffle(S)
#             E = 0
#             for ob in S:
#                 for i in range(R.shape[0]):
#                     for j in range(R.shape[1]):
#                         e = R[i][j] - U[i][q] * V[j][q]
#                         E += e ** 2
#                         utmp = U[i][q] + alpha * (e * V[j][q] - lambda_v * U[i][q])
#                         vtmp = V[j][q] + alpha * (e * U[i][q] - lambda_v * V[j][q])
#                         U[i][q] = utmp
#                         V[j][q] = vtmp
#             epoch_new = 0.5 * E + lambda_v * 0.5 * (U ** 2).sum() + lambda_v * 0.5 * (V ** 2).sum()
#             convergence = (epoch_new < epoch)
#             print('epoch=%s' % epoch_new)
#         for ob2 in S:
#             for i in range(R.shape[0]):
#                 for j in range(R.shape[1]):
#                     R[i][j] = R[i][j]
#     return U, V


def als(R, k, steps=1000):
    R_shape = R.shape
    U = np.random.rand(k, R_shape[0])
    I = np.random.rand(k, R_shape[1])
    epoch_U = []
    epoch_I = []
    for step in range(steps):
        epu = 0
        epi = 0
        for a in range(U.shape[1]):
            epu += ((np.fabs(U[:, a] - np.linalg.inv(I.dot(I.T)).dot(I).dot(R[a].T))).sum())
            U[:, a] = np.linalg.inv(I.dot(I.T)).dot(I).dot(R[a].T)
        for b in range(I.shape[1]):
            epi += ((np.fabs(I[:, b] - np.linalg.inv(U.dot(U.T)).dot(U).dot(R[:, b].T))).sum())
            I[:, b] = np.linalg.inv(U.dot(U.T)).dot(U).dot(R[:, b].T)
        epoch_U.append(epu / U.shape[1])
        epoch_I.append(epi / I.shape[1])
        print('\r%s' % step, end='', flush=True)
    return U, I, epoch_U, epoch_I


def matrix_preparation(s_rate, k, steps=1000):
    U, I, epoch_U, epoch_I = als(s_rate.fillna(0).values, k, steps)
    NewData = U.T.dot(I)
    ND_DF = pd.DataFrame(NewData, index=s_rate.index, columns=s_rate.columns)
    # 将原矩阵评分添加进去
    # s_rate_predict = (
    #         s_rate_mean.fillna(0) + ND_DF[s_rate_mean.isnull()].fillna(0))
    # 如果只看近似值的话就使用这个
    s_rate_predict = ND_DF
    return s_rate_predict, epoch_U, epoch_I


def recommend_als(s_rate_mean, s_rate_predict, s_similar, u, num=10, similar_limit=0.8):
    """
    生成推荐列表

    :param s_rate_mean:    平均化后的用户评分矩阵
    :param s_rate_predict: 预测用户评分矩阵
    :param s_similar:      项目相似度矩阵
    :param u:              用户ID：整数
    :param num:            列表内最大项数：整数，默认为10
    :param similar_limit:  相似度的最大值：浮点数，默认为0.8
    :return: 推荐列表：列表，项为元组，元组内项分别为项目ID、项目的评分
    """
    # 选取原评分矩阵中未评分的，且预测值在平均评分及以上的项目
    r_1 = s_rate_predict.loc[s_rate_mean[u].isnull(), u][
        s_rate_predict[u] >= 0].sort_values(ascending=False)
    r_2 = r_1
    # 选取原评分矩阵中评分的，且预测值在平均评分及以上的项目
    r_11 = s_rate_predict.loc[~s_rate_mean[u].isnull(), u][
        s_rate_predict[u] >= 0].sort_values(ascending=False)
    # 筛选相对于已评分项目，相似度低的项目
    for j in r_11.index:
        s_s_1 = s_similar.loc[s_similar.loc[j] < similar_limit, j]
        r_2 = r_2[r_2.index & s_s_1.index]
    rec_movies = [([i, r_1[i]]) for i in r_1[r_2.index & r_1.index][:num].index]
    return rec_movies


if __name__ == '__main__':
    K = 40
    STEPS = 1000
    RECOMMEND_NUM = 10
    SIMILAR_LIMIT = 0.8

    s_rate = pd.read_csv('../temp/s_rate.csv')
    s_rate = s_rate.set_index('MovieID')
    s_rate.rename(columns=int, inplace=True)

    s_similar = pd.read_csv('../temp/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)

    s_rate_mean = s_rate_mean(s_rate)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
          '任务开始')
    s_rate_predict, epoch_U, epoch_I = matrix_preparation(s_rate_mean, K, STEPS)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
          '任务完成')
    rl = recommend_als(s_rate_mean, s_rate_predict, s_similar, 1)
    print(rl)

    plt.plot(list(range(STEPS)), epoch_U, color='r')
    plt.plot(list(range(STEPS)), epoch_I, color='g')
    plt.show()

    rmse = np.sqrt(
        ((s_rate_mean.fillna(0) - s_rate_predict[
            ~s_rate_mean.isnull()]) ** 2).sum().sum() / (
            s_rate_mean.count().count()))
    print('RMSE: %s' % rmse)

    # 覆盖率 Coverage & 多样性 Diversity
    ru_set = set()
    sum_diversity_u = 0
    for u in s_rate.columns:
        # recommend_list_with_score = recommend_svd(s_rate_mean, s_rate_predict, s_similar, u)
        recommend_list_with_score = recommend_als(s_rate_mean, s_rate_predict, s_similar, u, RECOMMEND_NUM, SIMILAR_LIMIT)
        recommend_list = [i[0] for i in recommend_list_with_score]
        sum_diversity_u += 1 - (s_similar.loc[recommend_list, recommend_list
                                ].sum().sum() - len(
            recommend_list)) / (0.5 * len(recommend_list) * (
                len(recommend_list) - 1))
        for i in recommend_list:
            ru_set.add(i)
        print('\r%s' % u, end='', flush=True)
    coverage = len(ru_set) / len(s_rate.index)
    diversity = sum_diversity_u / len(s_rate.columns)
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)
    # on 1/10 dataset
    # RMSE: 2.054283044022762
    # Coverage: 0.6649484536082474
    # Diversity: -0.2200382931728572