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


def als(R, k, steps=1000, epoch=0.00001):
    """
    执行ALS算法

    :param R: 待分解矩阵
    :param k:
    :param steps:
    :param epoch:
    :return:
    """
    R_shape = R.shape
    U = np.random.rand(k, R_shape[0])
    I = np.random.rand(k, R_shape[1])
    Us = []
    Is = []
    epoch_U = []
    epoch_I = []
    Us.append(U.copy())
    Is.append(I.copy())
    for step in range(steps):
        epu = 0
        epi = 0
        for a in range(U.shape[1]):
            epu += ((np.fabs(U[:, a] - np.linalg.inv(I.dot(I.T)).dot(I).dot(R[a].T))).sum())
            U[:, a] = np.linalg.inv(I.dot(I.T)).dot(I).dot(R[a].T)
        for b in range(I.shape[1]):
            epi += ((np.fabs(I[:, b] - np.linalg.inv(U.dot(U.T)).dot(U).dot(R[:, b].T))).sum())
            I[:, b] = np.linalg.inv(U.dot(U.T)).dot(U).dot(R[:, b].T)
        epu_avg = epu / U.shape[1]
        epi_avg = epi / I.shape[1]
        epoch_U.append(epu_avg)
        epoch_I.append(epi_avg)
        Us.append(U.copy())
        Is.append(I.copy())
        if epu_avg < epoch and epi_avg < epoch:
            break
        # print('\r%s' % step, end='', flush=True)
    epu = 0
    epi = 0
    for a in range(U.shape[1]):
        epu += ((np.fabs(
            U[:, a] - np.linalg.inv(I.dot(I.T)).dot(I).dot(R[a].T))).sum())
    for b in range(I.shape[1]):
        epi += ((np.fabs(
            I[:, b] - np.linalg.inv(U.dot(U.T)).dot(U).dot(R[:, b].T))).sum())
    epu_avg = epu / U.shape[1]
    epi_avg = epi / I.shape[1]
    epoch_U.append(epu_avg)
    epoch_I.append(epi_avg)
    return Us, Is, epoch_U, epoch_I


def s_rate_predict_gen(s_rate, U, I):
    NewData = U.T.dot(I)
    ND_DF = pd.DataFrame(NewData, index=s_rate.index, columns=s_rate.columns)
    # 将原矩阵评分添加进去
    # s_rate_predict = (
    #         s_rate_mean.fillna(0) + ND_DF[s_rate_mean.isnull()].fillna(0))
    # 如果只看近似值的话就使用这个
    s_rate_predict = ND_DF
    return s_rate_predict


def matrix_preparation(s_rate, k, steps=1000, epoch=0.00001):
    Us, Is, epoch_U, epoch_I = als(s_rate.fillna(0).values, k, steps, epoch)
    # NewData = Us[-1].T.dot(Is[-1])
    # ND_DF = pd.DataFrame(NewData, index=s_rate.index, columns=s_rate.columns)
    # 将原矩阵评分添加进去
    # s_rate_predict = (
    #         s_rate_mean.fillna(0) + ND_DF[s_rate_mean.isnull()].fillna(0))
    # 如果只看近似值的话就使用这个
    #s_rate_predict = ND_DF
    s_rate_predict = s_rate_predict_gen(s_rate, Us[-1], Is[-1])
    return s_rate_predict, Us, Is, epoch_U, epoch_I


def recommend_als(s_rate_mean, s_rate_predict, u, num=10):
    """
    生成推荐列表

    :param s_rate_mean:    平均化后的用户评分矩阵
    :param s_rate_predict: 预测用户评分矩阵
    :param u:              用户ID：整数
    :param num:            列表内最大项数：整数，默认为10
    :return: 推荐列表：列表，项为元组，元组内项分别为项目ID、项目的评分
    """
    # 选取原评分矩阵中未评分的，且预测值在平均评分及以上的项目
    r_1 = s_rate_predict.loc[s_rate_mean[u].isnull(), u][
        s_rate_predict[u] >= 0].sort_values(ascending=False)
    rec_movies = [([i, r_1[i]]) for i in r_1[:num].index]
    return rec_movies


def recommend_als_s(s_rate_mean, s_rate_predict, s_similar, u, num=10, similar_limit=0.8):
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
    K = 80
    STEPS = 1000
    RECOMMEND_NUM = 10
    EPOCH = 0.0001
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
    model_start = time.time()  # 打点计时
    s_rate_predict, Us, Is, epoch_U, epoch_I = matrix_preparation(s_rate_mean, K, STEPS, EPOCH)
    model_end = time.time()  # 打点计时
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
          '任务完成')
    rl = recommend_als(s_rate_mean, s_rate_predict, 1)
    print(rl)
    print(model_end - model_start)  # 打点计时

    # plt.plot(list(range(len(epoch_U))), epoch_U, color='r')
    # plt.plot(list(range(len(epoch_U))), epoch_I, color='g')
    # plt.show()

    # steps_list = []
    # RMSEs = []
    # Covs = []
    # Divs = []
    # for ii, (U, I) in enumerate(zip(Us, Is)):
    #     print('='*20, '第%s次迭代' % ii, '='*20)
    #     s_rate_predict = s_rate_predict_gen(s_rate, U, I)
    #     rmse = np.sqrt(
    #         ((s_rate_mean.fillna(0) - s_rate_predict[
    #             ~s_rate_mean.isnull()]) ** 2).sum().sum() / (
    #             s_rate_mean.count().count()))
    #     print('RMSE: %s' % rmse)
    #
    #     # 覆盖率 Coverage & 多样性 Diversity
    #     ru_set = set()
    #     sum_diversity_u = 0
    #     for u in s_rate.columns:
    #         # recommend_list_with_score = recommend_svd(s_rate_mean, s_rate_predict, u)
    #         recommend_list_with_score = recommend_als(s_rate_mean,
    #                                                   s_rate_predict,
    #                                                   u, RECOMMEND_NUM)
    #         recommend_list = [i[0] for i in recommend_list_with_score]
    #         sum_diversity_u += 1 - (s_similar.loc[recommend_list, recommend_list
    #                                 ].sum().sum() - len(
    #             recommend_list)) / (0.5 * len(recommend_list) * (
    #                 len(recommend_list) - 1))
    #         for i in recommend_list:
    #             ru_set.add(i)
    #         print('\r%s' % u, end='', flush=True)
    #     coverage = len(ru_set) / len(s_rate.index)
    #     diversity = sum_diversity_u / len(s_rate.columns)
    #     steps_list.append(ii)
    #     RMSEs.append(rmse)
    #     Covs.append(coverage)
    #     Divs.append(diversity)
    #     print('Coverage: %s' % coverage)
    #     print('Diversity: %s' % diversity)
    #
    # plt.title('Epoch of U and I (k = %s, epoch = %s)' % (K, EPOCH))
    # plt.plot(list(range(len(epoch_U))), epoch_U, color='r', label='epoch_U')
    # plt.plot(list(range(len(epoch_U))), epoch_I, color='g', label='epoch_U')
    # plt.xlabel('Steps')
    # plt.legend()
    # plt.show()
    # plt.title('RMSE (k = %s, epoch = %s)' % (K, EPOCH))
    # plt.plot(steps_list, RMSEs, color='r', label='RMSE')
    # plt.xlabel('Steps')
    # plt.legend()
    # plt.show()
    # plt.title('Coverage and Diversity (k = %s, epoch = %s)' % (K, EPOCH))
    # plt.plot(steps_list, Covs, color='r', label='Coverage')
    # plt.plot(steps_list, Divs, color='g', label='Diversity')
    # plt.xlabel('Steps')
    # plt.legend()
    # plt.show()
    rmse = np.sqrt(
        ((s_rate_mean.fillna(0) - s_rate_predict[
            ~s_rate_mean.isnull()]) ** 2).sum().sum() / (
            s_rate_mean.count().count()))
    print('RMSE: %s' % rmse)

    # 覆盖率 Coverage & 多样性 Diversity
    ru_set = set()
    sum_diversity_u = 0
    rec_time_sum = 0  # 打点计时
    for u in s_rate.columns:
        rec_start = time.time()  # 打点计时
        # recommend_list_with_score = recommend_svd(s_rate_mean, s_rate_predict, u)
        recommend_list_with_score = recommend_als(s_rate_mean, s_rate_predict, u, RECOMMEND_NUM)
        rec_end = time.time()  # 打点计时
        recommend_list = [i[0] for i in recommend_list_with_score]
        sum_diversity_u += 1 - (s_similar.loc[recommend_list, recommend_list
                                ].sum().sum() - len(
            recommend_list)) / (0.5 * len(recommend_list) * (
                len(recommend_list) - 1))
        for i in recommend_list:
            ru_set.add(i)
        rec_time_sum += rec_end - rec_start  # 打点计时
        print('\r%s' % u, end='', flush=True)
    coverage = len(ru_set) / len(s_rate.index)
    diversity = sum_diversity_u / len(s_rate.columns)
    rec_time = rec_time_sum / len(s_rate.columns)  # 打点计时
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)
    print('Time: %s' % rec_time)  # 打点计时

    # on 1/10 dataset(k=40, 1000)
    # RMSE: 2.054283044022762
    # Coverage: 0.6649484536082474
    # Diversity: -0.2200382931728572

    # k=20
    # RMSE: 2.760603005878657
    # Coverage: 0.6494845360824743
    # Diversity: -0.23050609756750576

    # k=80
    # RMSE: 1.196139968064099
    # Coverage: 0.7242268041237113
    # Diversity: -0.22817923964866554