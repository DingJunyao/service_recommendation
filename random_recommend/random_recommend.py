#!/usr/bin/python3
"""
随机推荐（对照组）

Author: DingJunyao
Date: 2019-05-14 21:08
"""

import pandas as pd
import random
import time


def random_recommend(items, num=10):
    """
    随机返回特定项数的电影推荐列表

    :param items: 电影ID的列表
    :param num:   项数，默认为10
    :return: 电影ID列表
    """
    random.shuffle(items)
    return items[:num]


if __name__ == '__main__':
    MATRIX_PATH = '../temp'
    s_rate_old = pd.read_csv(MATRIX_PATH + '/s_rate_old.csv')
    s_rate_old = s_rate_old.set_index('MovieID')
    s_rate_old.rename(columns=int, inplace=True)

    model_start = time.time()  # 打点计时
    items = list(s_rate_old.index)
    model_end = time.time()  # 打点计时
    recommend_list_example = random_recommend(items)
    print(recommend_list_example)
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
        select_set = set(s_rate_new[~s_rate_new[u].isnull()][u].index)
        rec_start = time.time()  # 打点计时
        recommend_list = random_recommend(items)
        rec_end = time.time()  # 打点计时
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
