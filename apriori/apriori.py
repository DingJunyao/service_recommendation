#!/usr/bin/python3
"""
通过Apriori算法进行关联规则挖掘

Author: DingJunyao
Date: 2019-05-14 22:24
"""

import pickle
import pandas as pd
import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def recommend_apriori_series(series, association_rules, num=10):
    """
    通过给定序列和关联规则，返回特定项数的推荐物品

    :param series:            序列：列表，元素为电影ID，一般是最近评分的电影
    :param association_rules: 关联规则：pandas.DataFrame，
                                为mlxtend.frequent_patterns.association_rules
                                生成的关联规则
    :param num:               项数，默认为10
    :return: 推荐列表：每一项为元组，包括电影ID和置信度
    """
    if not series:
        # print('No series')
        return []
    recommend_list = []
    for i in association_rules.itertuples():
        if i[1].issubset(series):
            recommend_list.append((i[2], i[6]))
    recommend_list.sort(key=lambda x: x[1], reverse=True)
    recommend_list_out = {}
    for i in recommend_list:
        for j in list(i[0]):
            if j not in recommend_list:
                recommend_list_out[j] = i[1]
    recommend_list_out = list(recommend_list_out.items())
    recommend_list_out.sort(key=lambda x: x[1], reverse=True)
    if recommend_list_out:
        return recommend_list_out[:num]
    else:
        # print('No match')
        return []


def recommend_apriori(ratings, association_rules, uid, num=10,
                      time_limit=0.2):
    """
    给定评价、评价时间和关联规则，根据用户最近评价的电影，为其推荐若干关联电影
    是对recommend_apriori_series的封装

    :param ratings:           评分表
    :param association_rules: 关联规则
    :param uid:               用户ID
    :param num:               项数，默认为10
    :param time_limit:        确定最近的评价记录时，确定最近若干比例的电影
                                默认为0.2
    :return: 推荐列表
    """
    time_limit_reverse = 1 - time_limit
    ratings_u = ratings[(ratings['UserID'] == uid)]
    recent_items = ratings_u[(
                ratings_u['Timestamp'] > ratings_u['Timestamp'].quantile(
            time_limit_reverse))]['MovieID']
    recent_items_list = list(recent_items)
    return recommend_apriori_series(recent_items_list, association_rules, num)


if __name__ == '__main__':
    ML_DS_PATH = '../dataset/ml-out'
    MATRIX_PATH = '../temp'

    ratings_old = pd.read_csv(ML_DS_PATH + '/ratings_old.csv')

    s_rate_old = pd.read_csv(MATRIX_PATH + '/s_rate_old.csv')
    s_rate_old = s_rate_old.set_index('MovieID')
    s_rate_old.rename(columns=int, inplace=True)

    with open(MATRIX_PATH + '/cluster_old.pickle', 'rb') as f:
        cluster = pickle.load(f)

    model_start = time.time()  # 打点计时
    te = TransactionEncoder()
    te_ary = te.fit(cluster).transform(cluster)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.09, use_colnames=True)
    ar = association_rules(frequent_itemsets, metric="confidence",
                           min_threshold=0.2)
    model_end = time.time()  # 打点计时
    print('Modeling Time: %s' % (model_end - model_start))  # 打点计时
    recommend_list_example = recommend_apriori(ratings_old, ar, 5)
    print(recommend_list_example)

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
        recommend_list_with_score = recommend_apriori(ratings_old, ar, u)
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
