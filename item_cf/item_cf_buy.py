#!/usr/bin/python3
"""
基于物品同时购买频率为权重指标的协同过滤

Author: DingJunyao
Date: 2019-05-15 16:05
"""

import pandas as pd
import numpy as np
import pickle
import itertools
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


def item_similar_on_buy(item_buy_number, s_bought, i, j):
    """
    计算物品之间在购买上的相似度

    :param item_buy_number: 物品被购买次数矩阵
    :param s_bought:        物品同时被购买的次数的矩阵
    :param i:               物品i的ID
    :param j:               物品j的ID
    :return: 物品同时购买次数矩阵
    """
    if item_buy_number[i] * item_buy_number[j] != 0:
        w_ij = s_bought.loc[i, j] / np.sqrt(item_buy_number[i] * item_buy_number[j])
    else:
        w_ij = 0.0
    return w_ij


def s_bought_gen(cluster, s_rate):
    """
    生成物品同时被购买的次数的矩阵

    :param cluster: 聚类
    :param s_rate:  评分矩阵
    :return: 物品同时被购买的次数的矩阵
    """
    s_bought = pd.DataFrame(columns=s_rate.index, index=s_rate.index,
                            dtype=np.int).fillna(0)
    for ii, i in enumerate(cluster):
        for j in itertools.combinations(i, 2):
            s_bought.loc[j[0], j[1]] += 1
            s_bought.loc[j[1], j[0]] += 1
        print('\r%s' % ii, end='', flush=True)
    return s_bought


def s_similar_on_buy_gen(s_rate, s_bought):
    """
    生成物品在购买上的相似度的矩阵

    :param s_rate:   评分矩阵
    :param s_bought: 物品同时被购买的次数的矩阵
    :return: 物品在购买上的相似度的矩阵
    """
    s_similar_on_buy = pd.DataFrame(columns=s_rate.index, index=s_rate.index,
                                    dtype=np.float)
    item_buy_number = s_rate.T.count()
    for ii, i in enumerate(s_similar_on_buy.index):
        s_similar_on_buy.loc[i, i] = 1
        for j in s_similar_on_buy.index[ii + 1:]:
            s_similar_on_buy.loc[i, j] = item_similar_on_buy(item_buy_number, s_bought, i, j)
            s_similar_on_buy.loc[j, i] = s_similar_on_buy.loc[i, j]
        print('\r%s' % ii, end='', flush=True)
    return s_similar_on_buy


def recent_items_df_gen(s_rate, rate_time):
    recent_items_df = s_rate.copy()
    recent_items_df[~(rate_time.max() - rate_time <= 365 * 86400)] = np.nan
    recent_items_df = recent_items_df.mask(recent_items_df >= 0, 1)
    return recent_items_df


def recent_items_gen(recent_items_df, u):
    return list(recent_items_df[u][~recent_items_df[u].isnull()].index)


def matrix_prepare(s_rate, s_similar_on_buy):
    """
    为协同过滤进行矩阵准备

    :param s_rate:           评分表：pandas.DataFrame
    :param s_similar_on_buy: 各项目之间在购买上的相似性：pandas.DataFrame
    :return: 预测的评分表：pandas.DataFrame
    """
    s_predict = pd.DataFrame(index=s_rate.index, columns=s_rate.columns, dtype=np.float).fillna(0)
    s_predict = s_predict.fillna(0) + s_similar_on_buy.fillna(0).values.dot(s_rate.fillna(0).values)
    return s_predict

def recommend_icf(s_predict, s_rate, u, num=10):
    """
    返回推荐结果

    :param s_predict: 预测评分矩阵
    :param s_rate:    评分矩阵
    :param u:         用户ID
    :param num:       项数，默认为i0
    :return: 推荐列表
    """
    return [
        (k, s_predict.loc[k, u])
        for k in s_predict[~s_rate[u].isnull()][u].sort_values(ascending=False).index[:num]
    ]


if __name__ == '__main__':
    ML_DS_PATH = '../dataset/ml-out'
    MATRIX_PATH = '../temp'

    s_rate_old = pd.read_csv(MATRIX_PATH + '/s_rate_old.csv')
    s_rate_old = s_rate_old.set_index('MovieID')
    s_rate_old.rename(columns=int, inplace=True)

    s_rate_equalized, s_rate_old_mean= s_rate_equalization(s_rate_old)

    with open(MATRIX_PATH + '/cluster_old.pickle', 'rb') as f:
        cluster = pickle.load(f)

    model_start = time.time()  # 打点计时
    s_bought = s_bought_gen(cluster, s_rate_old)
    s_similar_on_buy = s_similar_on_buy_gen(s_rate_old, s_bought)
    s_rate_predict = matrix_prepare(s_rate_equalized, s_similar_on_buy)
    model_end = time.time()  # 打点计时
    recommend_item = recommend_icf(s_rate_predict, s_rate_equalized, 5, num=10)
    print(recommend_item)
    print('Modeling Time: %s' % (model_end - model_start))  # 打点计时

    # 准确率 Precision & 召回率 Recall & 覆盖率 Coverage & 多样性 Diversity
    s_rate_new = pd.read_csv(MATRIX_PATH + '/s_rate_new.csv')
    s_rate_new = s_rate_new.set_index('MovieID')
    s_rate_new.rename(columns=int, inplace=True)

    s_similar = pd.read_csv(MATRIX_PATH + '/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)


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
        recommend_list_with_score = recommend_icf(s_rate_predict, s_rate_equalized, u)
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
    print('Precision: %s' % precision)
    print('Recall: %s' % recall)
    print('F-Measure: %s' % f_measure)
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)
