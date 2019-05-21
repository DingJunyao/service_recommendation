#!/usr/bin/python3
"""


Author: DingJunyao
Date: 2019-05-13 16:54
"""

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def s_rate_equalization(s_rate):
    """
    平均化矩阵内各用户评分

    :param s_rate: 用户评分矩阵：pandas.DataFrame
    :return:       平均化后的用户评分矩阵：pandas.DataFrame
                   各用户的评分的平均值：pandas.Series
    """
    s_rate_mean = s_rate.mean()
    return (s_rate - s_rate.mean()), s_rate_mean


def als(R, k, steps=1000, epoch=0.00001, lambda_u=0.0, lambda_i=0.0):
    """
    执行ALS算法

    :param R: 待分解矩阵
    :param k:
    :param steps:
    :param epoch:
    :return:
    """
    R_T = R.T   # n*m
    R_T_shape = R_T.shape
    U = np.random.rand(R_T_shape[0], k)   # n*k
    I = np.random.rand(R_T_shape[1], k)   # m*k
    Us = []
    Is = []
    ep = []
    ep_s = ((R_T - U.dot(I.T)) ** 2).sum() \
           + lambda_u * ((np.linalg.norm(U, axis=1) ** 2).sum()) \
           + lambda_i * ((np.linalg.norm(I, axis=1) ** 2).sum())
    ep.append(ep_s)
    Us.append(U.copy())
    Is.append(I.copy())
    for step in range(steps):
        for u in range(U.shape[0]):
            U[u] = (R_T[u].dot(I)).dot(np.linalg.inv(I.T.dot(I) + lambda_u * np.eye(k)))
        for i in range(I.shape[1]):
            I[i] = R_T[:, i].dot(U).dot(np.linalg.inv(U.T.dot(U) + lambda_i * np.eye(k)))
        ep_s = ((R_T - U.dot(I.T)) ** 2).sum()\
               + lambda_u * ((np.linalg.norm(U, axis=1) ** 2).sum())\
               + lambda_i * ((np.linalg.norm(I, axis=1) ** 2).sum())
        Us.append(U.copy())
        Is.append(I.copy())
        ep.append(ep_s)
        print('\r%s\t%s' % (step, ep_s), end='', flush=True)
        if ep_s < epoch:
            break
    return Us, Is, ep



def s_rate_predict_gen(s_rate, U, I):
    # NewData = U.T.dot(I)
    NewData = U.dot(I.T).T
    ND_DF = pd.DataFrame(NewData, index=s_rate.index, columns=s_rate.columns)
    # 将原矩阵评分添加进去
    # s_rate_predict = (
    #         s_rate_mean.fillna(0) + ND_DF[s_rate_mean.isnull()].fillna(0))
    # 如果只看近似值的话就使用这个
    s_rate_predict = ND_DF
    return s_rate_predict


def matrix_preparation(s_rate, k, steps=1000, epoch=0.00001, lambda_u=0.01, lambda_i=0.01):
    Us, Is, ep = als(s_rate.fillna(0).values, k, steps, epoch, lambda_u, lambda_i)
    # NewData = Us[-1].T.dot(Is[-1])
    # ND_DF = pd.DataFrame(NewData, index=s_rate.index, columns=s_rate.columns)
    # 将原矩阵评分添加进去
    # s_rate_predict = (
    #         s_rate_mean.fillna(0) + ND_DF[s_rate_mean.isnull()].fillna(0))
    # 如果只看近似值的话就使用这个
    #s_rate_predict = ND_DF
    s_rate_predict = s_rate_predict_gen(s_rate, Us[-1], Is[-1])
    return s_rate_predict, Us, Is, ep


def recommend_als(s_rate_equalized, s_rate_predict, u, num=10):
    """
    生成推荐列表

    :param s_rate_equalized: 平均化后的用户评分矩阵
    :param s_rate_predict:   预测用户评分矩阵
    :param u:                用户ID：整数
    :param num:              列表内最大项数：整数，默认为10
    :return: 推荐列表：列表，项为元组，元组内项分别为项目ID、项目的评分
    """
    # 选取原评分矩阵中未评分的，且预测值在平均评分及以上的项目
    r_1 = s_rate_predict.loc[s_rate_equalized[u].isnull(), u][
        s_rate_predict[u] >= 0].sort_values(ascending=False)
    rec_movies = [([i, r_1[i]]) for i in r_1[:num].index]
    return rec_movies


# def recommend_als_s(s_rate_mean, s_rate_predict, s_similar, u, num=10, similar_limit=0.8):
#     """
#     生成推荐列表
#
#     :param s_rate_mean:    平均化后的用户评分矩阵
#     :param s_rate_predict: 预测用户评分矩阵
#     :param s_similar:      项目相似度矩阵
#     :param u:              用户ID：整数
#     :param num:            列表内最大项数：整数，默认为10
#     :param similar_limit:  相似度的最大值：浮点数，默认为0.8
#     :return: 推荐列表：列表，项为元组，元组内项分别为项目ID、项目的评分
#     """
#     # 选取原评分矩阵中未评分的，且预测值在平均评分及以上的项目
#     r_1 = s_rate_predict.loc[s_rate_mean[u].isnull(), u][
#         s_rate_predict[u] >= 0].sort_values(ascending=False)
#     r_2 = r_1
#     # 选取原评分矩阵中评分的，且预测值在平均评分及以上的项目
#     r_11 = s_rate_predict.loc[~s_rate_mean[u].isnull(), u][
#         s_rate_predict[u] >= 0].sort_values(ascending=False)
#     # 筛选相对于已评分项目，相似度低的项目
#     for j in r_11.index:
#         s_s_1 = s_similar.loc[s_similar.loc[j] < similar_limit, j]
#         r_2 = r_2[r_2.index & s_s_1.index]
#     rec_movies = [([i, r_1[i]]) for i in r_1[r_2.index & r_1.index][:num].index]
#     return rec_movies


if __name__ == '__main__':
    K = 40
    STEPS = 100
    RECOMMEND_NUM = 10
    EPOCH = 1
    LAMBDA_U = 0.00001
    LAMBDA_I = 0.00001

    ML_DS_PATH = '../dataset/ml-out-sample'
    MATRIX_PATH = '../temp-sample'

    s_rate_old = pd.read_csv(MATRIX_PATH + '/s_rate_old.csv')
    s_rate_old = s_rate_old.set_index('MovieID')
    s_rate_old.rename(columns=int, inplace=True)

    s_similar = pd.read_csv(MATRIX_PATH + '/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)

    s_rate_old_equalized, s_rate_old_mean = s_rate_equalization(s_rate_old)
    model_start = time.time()  # 打点计时
    s_rate_predict, Us, Is, epo = matrix_preparation(s_rate_old_equalized, K, STEPS, EPOCH, LAMBDA_U, LAMBDA_I)
    model_end = time.time()  # 打点计时
    recommend_list_example = recommend_als(s_rate_old_equalized, s_rate_predict, 1)
    print(recommend_list_example)
    print('Modeling Time: %s' % (model_end - model_start))  # 打点计时

    # RMSE
    s_rate_new = pd.read_csv(MATRIX_PATH + '/s_rate_new.csv')
    s_rate_new = s_rate_new.set_index('MovieID')
    s_rate_new.rename(columns=int, inplace=True)

    s_rate_predict_restore = s_rate_predict + s_rate_old_mean.fillna(0)

    rmse = np.sqrt(((s_rate_new[~s_rate_new.isnull()] - (
    s_rate_predict_restore[~s_rate_new.isnull()])) ** 2).sum().sum() / (
                       (~s_rate_new.isnull()).sum().sum()))

    # 准确率 Precision & 召回率 Recall & 覆盖率 Coverage & 多样性 Diversity

    r_and_s_sum = 0
    r_sum = 0
    s_sum = 0
    ru_set = set()
    sum_diversity_u = 0
    user_minus = 0
    rec_time_sum = 0  # 打点计时
    for u in s_rate_old.columns:
        print('\r%s' % u, end='', flush=True)
        select_set = set(s_rate_new[~s_rate_new[u].isnull()][u].index)
        rec_start = time.time()  # 打点计时
        recommend_list_with_score = recommend_als(s_rate_old_equalized, s_rate_predict, u)
        rec_end = time.time()  # 打点计时
        recommend_list = [i[0] for i in recommend_list_with_score]
        recommend_set = set(recommend_list)
        r_and_s_sum += len(recommend_set & select_set)
        r_sum += len(recommend_set)
        s_sum += len(select_set)
        if len(recommend_list) > 1:
            sum_diversity_u += 1 - (
                    s_similar.loc[
                        recommend_list, recommend_list
                    ].sum().sum() - len(recommend_list)) / (
                                       0.5 * len(recommend_list) * (
                                       len(recommend_list) - 1))
        else:
            user_minus += 1
        for i in recommend_list:
            ru_set.add(i)
        rec_time_sum += rec_end - rec_start  # 打点计时
    coverage = len(ru_set) / len(s_rate_old.index)
    diversity = sum_diversity_u / (len(s_rate_old.columns) - user_minus)
    rec_time = rec_time_sum / len(s_rate_old.columns)  # 打点计时
    precision = r_and_s_sum / r_sum
    recall = r_and_s_sum / s_sum
    f_measure = (2 * precision * recall) / (precision + recall)
    print('Average Recommend Time: %s' % rec_time)  # 打点计时
    print('RMSE: %s' % rmse)
    print('Precision: %s' % precision)
    print('Recall: %s' % recall)
    print('F-Measure: %s' % f_measure)
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)

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



    # rmse = np.sqrt(
    #     ((s_rate_mean.fillna(0) - s_rate_predict[
    #         ~s_rate_mean.isnull()]) ** 2).sum().sum() / (
    #         s_rate_mean.count().count()))
    # print('RMSE: %s' % rmse)
    #
    # # 覆盖率 Coverage & 多样性 Diversity
    # ru_set = set()
    # sum_diversity_u = 0
    # rec_time_sum = 0  # 打点计时
    # for u in s_rate.columns:
    #     rec_start = time.time()  # 打点计时
    #     # recommend_list_with_score = recommend_svd(s_rate_mean, s_rate_predict, u)
    #     recommend_list_with_score = recommend_als(s_rate_mean, s_rate_predict, u, RECOMMEND_NUM)
    #     rec_end = time.time()  # 打点计时
    #     recommend_list = [i[0] for i in recommend_list_with_score]
    #     sum_diversity_u += 1 - (s_similar.loc[recommend_list, recommend_list
    #                             ].sum().sum() - len(
    #         recommend_list)) / (0.5 * len(recommend_list) * (
    #             len(recommend_list) - 1))
    #     for i in recommend_list:
    #         ru_set.add(i)
    #     rec_time_sum += rec_end - rec_start  # 打点计时
    #     print('\r%s' % u, end='', flush=True)
    # coverage = len(ru_set) / len(s_rate.index)
    # diversity = sum_diversity_u / len(s_rate.columns)
    # rec_time = rec_time_sum / len(s_rate.columns)  # 打点计时
    # print('Coverage: %s' % coverage)
    # print('Diversity: %s' % diversity)
    # print('Time: %s' % rec_time)  # 打点计时

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