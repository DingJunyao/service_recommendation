"""
基于SVD矩阵分解的推荐

Author: DingJunyao
Date：2019-05-17 18:24
"""

import numpy as np
import pandas as pd
import time


def s_rate_equalization(s_rate):
    """
    平均化矩阵内各用户评分

    :param s_rate: 用户评分矩阵：pandas.DataFrame
    :return:       平均化后的用户评分矩阵：pandas.DataFrame
                   各用户的评分的平均值：pandas.Series
    """
    s_rate_mean = s_rate.mean()
    return (s_rate - s_rate.mean()), s_rate_mean


def matrix_prepare(s_rate_equalized, energy_ratio=0.8):
    """
    准备评分预测矩阵：使用SVD分解

    :param s_rate_equalized:  平均化后的用户评分矩阵
    :param energy_ratio: 需要保留的能量比例：浮点数，默认为0.8
    :return: 预测用户评分矩阵
    """
    U, Sigma, VT = np.linalg.svd(s_rate_equalized.fillna(0).values)
    k = 0
    for i in range(len(Sigma)):
        if (Sigma[:i + 1] ** 2).sum() / (Sigma ** 2).sum() >= energy_ratio:
            k = i
            break
    # print(len(Sigma))
    # print(k)
    NewData = U[:, :k] * np.mat(np.eye(k) * Sigma[:k]) * VT[:k, :]
    ND_DF = pd.DataFrame(NewData, index=s_rate_equalized.index, columns=s_rate_equalized.columns)
    # 将原矩阵评分添加进去
    # s_rate_predict = (
    #         s_rate_equalized.fillna(0) + ND_DF[s_rate_equalized.isnull()].fillna(0))
    # 如果只看近似值的话就使用这个
    s_rate_predict = ND_DF
    return s_rate_predict


def recommend_svd(s_rate_mean, s_rate_predict, u, num=10):
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


def recommend_svd_s(s_rate_mean, s_rate_predict, s_similar, u, num=10, similar_limit=0.8):
    """
    生成推荐列表，排除近似项目

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


# def recommend_svd_time(s_rate_mean, s_rate_predict, s_similar, time_effect_rate, u, num=10,
#                        similar_limit=0.8, time_effect_limit=0.008):
#     """
#     生成推荐列表，考虑时间因素
#
#     :param s_rate_mean:       平均化后的用户评分矩阵
#     :param s_rate_predict:    预测用户评分矩阵
#     :param s_similar:         项目相似度矩阵
#     :param time_effect_rate:  时间效应矩阵
#     :param u:                 用户ID：整数
#     :param num:               列表内最大项数：整数，默认为10
#     :param similar_limit:     相似度的最大值：浮点数，默认为0.8
#     :param time_effect_limit: 时间效应的最大值：浮点数，默认为0.008
#     :return: 推荐列表
#     """
#     # 选取原评分矩阵中未评分的，且预测值在平均评分及以上的项目
#     r_1 = s_rate_predict.loc[s_rate_mean[u].isnull(), u][
#         s_rate_predict[u] >= 0].sort_values(ascending=False)
#     r_2 = r_1
#     # 选取原评分矩阵中最近评分的，且预测值在平均评分及以上的项目
#     r_11 = s_rate_predict.loc[~(s_rate_mean[u].isnull()) & (
#             time_effect_rate[u] >= time_effect_limit), u][
#         s_rate_predict[u] >= 0].sort_values(ascending=False)
#     # 筛选相对于已评分项目，相似度低的项目
#     for j in r_11.index:
#         s_s_1 = s_similar.loc[s_similar.loc[j] < similar_limit, j]
#         r_2 = r_2[r_2.index & s_s_1.index]
#     rec_movies = [([i, r_1[i]]) for i in r_1[r_2.index & r_1.index][:num].index]
#     return rec_movies


if __name__ == '__main__':
    ENERGY_RATIO = 0.8
    RECOMMEND_NUM = 10
    SIMILAR_LIMIT = 0.8
    TIME_EFFECT_LIMIT = 0.008

    MATRIX_PATH = '../temp-sample'
    s_rate_old = pd.read_csv(MATRIX_PATH + '/s_rate.csv')
    s_rate_old = s_rate_old.set_index('MovieID')
    s_rate_old.rename(columns=int, inplace=True)

    s_similar = pd.read_csv(MATRIX_PATH + '/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)

    s_rate_old_equalized, s_rate_old_meanaaaaaa = s_rate_equalization(s_rate_old)

    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '开始矩阵分解')
    model_start = time.time()  # 打点计时
    s_rate_predict = matrix_prepare(s_rate_old_equalized, ENERGY_RATIO)
    model_end = time.time()  # 打点计时
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
          '矩阵分解完成')
    # recommend_list_with_score = recommend_svd(s_rate_mean, s_rate_predict, 1)
    # recommend_list = [i[0] for i in recommend_list_with_score]
    recommend_list_with_score = recommend_svd(s_rate_old_equalized, s_rate_predict, 1)
    # recommend_svd_time(s_rate_mean, s_rate_predict, s_similar, time_effect_rate, 1)
    print('Modeling Time: %s' % (model_end - model_start))  # 打点计时
    # RMSE
    rmse = np.sqrt(
        ((s_rate_old_equalized.fillna(0) - s_rate_predict[
            ~s_rate_old_equalized.isnull()]) ** 2).sum().sum() / (
            s_rate_old_equalized.count().count()))
    print('RMSE: %s' % rmse)

    # 准确率 Precision & 召回率 Recall & 覆盖率 Coverage & 多样性 Diversity
    s_rate_new = pd.read_csv(MATRIX_PATH + '/s_rate_new.csv')
    s_rate_new = s_rate_new.set_index('MovieID')
    s_rate_new.rename(columns=int, inplace=True)

    s_similar = pd.read_csv(MATRIX_PATH + '/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)

    ru_set = set()
    sum_diversity_u = 0
    rec_time_sum = 0  # 打点计时
    for u in s_rate_old.columns:
        # recommend_list_with_score = recommend_svd(s_rate_mean, s_rate_predict, s_similar, u)
        rec_start = time.time()  # 打点计时
        recommend_list_with_score = recommend_svd(s_rate_old_equalized,
                                                       s_rate_predict,
                                                       u,
                                                       RECOMMEND_NUM,)
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
    coverage = len(ru_set) / len(s_rate_old.index)
    diversity = sum_diversity_u / len(s_rate.columns)
    rec_time = rec_time_sum / len(s_rate.columns)  # 打点计时
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)
    print('Time: %s' % rec_time)  # 打点计时

    # import time
    # er = []
    # tm = []
    # rm = []
    # cv = []
    # div = []
    # for i in range(60, 100):
    #     ENERGY_RATIO = i / 100
    #     RECOMMEND_NUM = 10
    #     SIMILAR_LIMIT = 0.8
    #     TIME_EFFECT_LIMIT = 0.008
    #     print('ENERGY_RATIO: %s' % ENERGY_RATIO)
    #     time_int = []
    #     for ti in range(5):
    #         time_start = time.time()
    #         s_rate_predict = matrix_prepare(s_rate_mean, ENERGY_RATIO)
    #         time_interval = time.time() - time_start
    #         time_int.append(time_interval)
    #     print('Time: %s' % (sum(time_int)/len(time_int)))
    #     tm.append(sum(time_int)/len(time_int))
    #
    #     # recommend_list_with_score = recommend_svd(s_rate_mean, s_rate_predict, 1)
    #     # recommend_list = [i[0] for i in recommend_list_with_score]
    #     # recommend_svd_time(s_rate_mean, s_rate_predict, s_similar, time_effect_rate, 1)
    #     # RMSE
    #     rmse = np.sqrt(
    #         ((s_rate_mean.fillna(0) - s_rate_predict[~s_rate_mean.isnull()]) ** 2).sum().sum() / (s_rate_mean.count().count()))
    #     print('RMSE: %s' % rmse)
    #
    #     # 覆盖率 Coverage & 多样性 Diversity
    #     ru_set = set()
    #     sum_diversity_u = 0
    #     for u in s_rate.columns:
    #         recommend_list_with_score = recommend_svd(s_rate_mean, s_rate_predict, u, RECOMMEND_NUM)
    #         # recommend_list_with_score = recommend_svd_time(s_rate_mean,
    #         #                                                s_rate_predict, s_similar,
    #         #                                                time_effect_rate, u,
    #         #                                                RECOMMEND_NUM,
    #         #                                                SIMILAR_LIMIT,
    #         #                                                TIME_EFFECT_LIMIT)
    #         recommend_list = [i[0] for i in recommend_list_with_score]
    #         sum_diversity_u += 1 - (s_similar.loc[
    #                                     recommend_list, recommend_list].sum().sum() - len(
    #             recommend_list)) / (0.5 * len(recommend_list) * (
    #                 len(recommend_list) - 1))
    #         for i in recommend_list:
    #             ru_set.add(i)
    #     coverage = len(ru_set) / len(s_rate.index)
    #     diversity = sum_diversity_u / len(s_rate.columns)
    #     print('Coverage: %s' % coverage)
    #     print('Diversity: %s' % diversity)
    #     print('-' * 40)
    #     er.append(ENERGY_RATIO)
    #     rm.append(rmse)
    #     cv.append(coverage)
    #     div.append(diversity)
    #
    #

    # import matplotlib.pyplot as plt
    # plt.plot(er, rm)
    # plt.plot(er, cv)
    # plt.plot(er, div)
    # plt.plot(er, tm)
    # plt.legend()
    # plt.show()
