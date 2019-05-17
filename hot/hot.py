#!/usr/bin/python3
"""
仅依靠热门内容进行推荐（对照组）

Author: DingJunyao
Date: 2019-05-12 22:17
"""

import pandas as pd
import time


def get_hot_items(s_rate):
    """
    返回项目按热度进行排序的结果

    :param s_rate: 评分矩阵
    :return: 热门项目列表：每一项为元组，包括电影ID和热度（总评分）
    """
    s_rate_hot_df = (s_rate.sum(axis=1)).sort_values(
        ascending=False)
    return [(i, s_rate_hot_df[i]) for i in s_rate_hot_df.index]


def recommend_hot(s_rate_hot, num=10):
    """
    返回特定项数的推荐列表

    :param s_rate_hot: 热门项目列表
    :param num:        项数，默认为10
    :return: 电影ID列表
    """
    return s_rate_hot[:num]


if __name__ == '__main__':
    MATRIX_PATH = '../temp-sample'
    s_rate_old = pd.read_csv(MATRIX_PATH + '/s_rate_old.csv')
    s_rate_old = s_rate_old.set_index('MovieID')
    s_rate_old.rename(columns=int, inplace=True)

    model_start = time.time()  # 打点计时
    s_rate_hot = get_hot_items(s_rate_old)
    recommend_list_with_score = recommend_hot(s_rate_hot)
    recommend_list = [i[0] for i in recommend_list_with_score]
    model_end = time.time()  # 打点计时
    print(recommend_list)
    print('Modeling Time: %s' % (model_end - model_start))  # 打点计时

    # 准确率 Precision & 召回率 Recall
    s_rate_new = pd.read_csv(MATRIX_PATH + '/s_rate_new.csv')
    s_rate_new = s_rate_new.set_index('MovieID')
    s_rate_new.rename(columns=int, inplace=True)

    s_similar = pd.read_csv(MATRIX_PATH + '/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)

    r_and_s_sum = 0
    r_sum = 0
    s_sum = 0
    for u in s_rate_old.columns:
        select_set = set(s_rate_new[~s_rate_new[u].isnull()][u].index)
        recommend_set = set(recommend_list)
        r_and_s_sum += len(recommend_set & select_set)
        r_sum += len(recommend_set)
        s_sum += len(select_set)
    precision = r_and_s_sum / r_sum
    recall = r_and_s_sum / s_sum
    f_measure = (2 * precision * recall) / (precision + recall)
    print('Precision: %s' % precision)
    print('Recall: %s' % recall)
    print('F-Measure: %s' % f_measure)

    # 覆盖率 Coverage
    ru_set = set()
    for i in recommend_list:
        ru_set.add(i)
    coverage = len(ru_set) / len(s_rate_old.index)
    print('Coverage: %s' % coverage)

    # 多样性 Diversity

    diversity = 1 - (
                s_similar.loc[recommend_list, recommend_list].sum().sum() - len(
                    recommend_list)) / (0.5 * len(recommend_list) * (
                        len(recommend_list) - 1))
    print('Diversity: %s' % diversity)

