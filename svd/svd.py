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
    NewData = U[:, :k] * np.mat(np.eye(k) * Sigma[:k]) * VT[:k, :]
    ND_DF = pd.DataFrame(NewData, index=s_rate_equalized.index,
                         columns=s_rate_equalized.columns)
    # 将原矩阵评分添加进去
    # s_rate_predict = (
    #         s_rate_equalized.fillna(0) + ND_DF[
    #             s_rate_equalized.isnull()].fillna(0))
    # 如果只看近似值的话就使用这个
    s_rate_predict = ND_DF
    return s_rate_predict


def recommend_svd(s_rate_equalized, s_rate_predict, u, num=10):
    """
    生成推荐列表

    :param s_rate_equalized: 平均化后的用户评分矩阵
    :param s_rate_predict:     预测用户评分矩阵
    :param u:                  用户ID：整数
    :param num:                列表内最大项数：整数，默认为10
    :return: 推荐列表：列表，项为元组，元组内项分别为项目ID、项目的评分
    """
    # 选取原评分矩阵中未评分的，且预测值在平均评分及以上的项目
    r_1 = s_rate_predict.loc[s_rate_equalized[u].isnull(), u][
        s_rate_predict[u] >= 0].sort_values(ascending=False)
    rec_movies = [([i, r_1[i]]) for i in r_1[:num].index]
    return rec_movies


if __name__ == '__main__':
    ENERGY_RATIO = 0.8
    RECOMMEND_NUM = 10

    ML_DS_PATH = '../dataset/ml-out-sample'
    MATRIX_PATH = '../temp-sample'

    s_rate_old = pd.read_csv(MATRIX_PATH + '/s_rate_old.csv')
    s_rate_old = s_rate_old.set_index('MovieID')
    s_rate_old.rename(columns=int, inplace=True)
    s_similar = pd.read_csv(MATRIX_PATH + '/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)

    s_rate_old_equalized, s_rate_old_mean = s_rate_equalization(s_rate_old)

    print('=' * 20 + 'ENERGY_RATIO=%s' % ENERGY_RATIO + '=' * 20)
    model_start = time.time()  # 打点计时
    s_rate_predict = matrix_prepare(s_rate_old_equalized, ENERGY_RATIO)
    model_end = time.time()  # 打点计时
    recommend_list_example = recommend_svd(s_rate_old_equalized, s_rate_predict,
                                           1)
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
        recommend_list_with_score = recommend_svd(s_rate_old_equalized,
                                                  s_rate_predict, u)
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
