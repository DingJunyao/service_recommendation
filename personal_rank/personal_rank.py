#!/usr/bin/python3
"""


Author: DingJunyao
Date: 2019-05-13 18:22
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


def graph_gen(s_rate, rate_limit=0):
    """
    生成二分图

    :param s_rate:     用户评分矩阵
    :param rate_limit: 评分不小于该值，才考虑该联系，默认值为0
    :return: 二分图：矩阵
             坐标轴上的值的名称

    """
    user_num = len(s_rate.columns)
    vertex = ['U_%s' % i for i in s_rate.columns]
    vertex.extend(['S_%s' % i for i in s_rate.index])
    graph = np.zeros((len(vertex), len(vertex)))
    for ii, i in enumerate(s_rate.index):
        for ji, j in enumerate(s_rate.columns):
            if s_rate.loc[i, j] >= rate_limit:
                graph[ii + user_num][ji] = 1
                graph[ji][ii + user_num] = 1
            print('\r%s\t%s' % (i, j), end='', flush=True)
    return graph, vertex


def M_gen(graph):
    """
    生成转移矩阵

    :param graph: 二分图
    :return: 转移矩阵
    """
    g = graph.copy().astype(np.float)
    # 防止除以0的情况
    g = g / np.where(g.sum(axis=0) == 0, 1, g.sum(axis=0)).reshape(-1, 1)
    return g


def matrix_prepare(s_rate, rate_limit=0, alpha=0.8):
    """
    矩阵准备，是上面两个函数的封装

    :param s_rate:     用户评分矩阵
    :param rate_limit: 评分不小于该值，才考虑该联系，默认值为0
    :param alpha:      节点上游走到下一个节点的概率，一般在0.6-0.8，默认为0.8
    :return: PersonalRank排序矩阵
             坐标轴上的值的名称
    """
    graph, vertex = graph_gen(s_rate, rate_limit)
    M = M_gen(graph)
    r_all = np.linalg.inv(np.eye(M.shape[0]) - alpha * M.T)
    return r_all, vertex


def scores_prepare(r_all, vertex):
    """
    PersonalRank排序表准备

    :param r_all:  PersonalRank排序矩阵
    :param vertex: 坐标轴上的值的名称
    :return: PersonalRank排序表
    """
    scores = pd.DataFrame(r_all, index=vertex, columns=vertex)
    scores = scores[scores.index.str.startswith('U_')]
    scores = scores.T[scores.T.index.str.startswith('S_')]
    scores.rename(index=lambda x: int(x[2:]), columns=lambda x: int(x[2:]),
                  inplace=True)
    return scores


def personal_rank(scores, u, num=10):
    """
    根据排序表进行推荐

    :param scores: 排序表
    :param u:      用户ID
    :param num:    项数
    :return: 推荐列表：每一项为元组，包括电影ID和排序用PR值
    """
    scores_u = scores[u].sort_values(ascending=False)
    rec_list = [([i, scores_u[i]]) for i in scores_u[:num].index]
    return rec_list


if __name__ == '__main__':
    ML_DS_PATH = '../dataset/ml-out-sample'
    MATRIX_PATH = '../temp-sample'

    s_rate_old = pd.read_csv(MATRIX_PATH + '/s_rate_old.csv')
    s_rate_old = s_rate_old.set_index('MovieID')
    s_rate_old.rename(columns=int, inplace=True)

    s_rate_old_equalized, s_rate_old_mean = s_rate_equalization(s_rate_old)

    model_start = time.time()  # 打点计时
    r, v = matrix_prepare(s_rate_old_equalized)
    scores = scores_prepare(r, v)
    model_end = time.time()  # 打点计时
    recommend_list_example = personal_rank(scores, 5)
    print(recommend_list_example)
    print('Model time: %s' % (model_end - model_start))  # 打点计时

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
        recommend_list_with_score = personal_rank(scores, u)
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
