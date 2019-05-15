#!/usr/bin/python3
"""


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
    if not series:
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
        return []


def recommend_apriori(ratings, uid, association_rules, num=10):
    ratings_u = ratings[(ratings['UserID'] == uid)]
    recent_items = ratings_u[
        (ratings_u['Timestamp'].max() - ratings_u['Timestamp'] <= 30 * 86400)][
        'MovieID']
    recent_items_list = list(recent_items.values)
    return recommend_apriori_series(recent_items_list, association_rules, num)


if __name__ == '__main__':
    ratings = pd.read_csv('../dataset/ml-out-reduce/ratings.csv')
    with open('../temp_ori/cluster.pickle', 'rb') as f:
        cluster = pickle.load(f)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
          '任务开始')
    te = TransactionEncoder()
    te_ary = te.fit(cluster).transform(cluster)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
    print(frequent_itemsets)
    ar = association_rules(frequent_itemsets, metric="confidence",
                           min_threshold=0.2)
    print(ar)
    # [1, 243, 23, 65]
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
          '提取关联规则完成')
    # ar_sel = ar[ar['confidence'] >= 0.6]
    print(recommend_apriori(ratings, 23, ar, num=10))

    # 覆盖率 Coverage & 多样性 Diversity
    s_rate = pd.read_csv('../temp_ori/s_rate.csv')
    s_rate = s_rate.set_index('MovieID')
    s_rate.rename(columns=int, inplace=True)

    s_similar = pd.read_csv('../temp_ori/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)
    ru_set = set()
    user_minus = 0
    sum_diversity_u = 0
    for u in s_rate.columns:
        recommend_list_with_score = recommend_apriori(ratings, u, ar)
        recommend_list = [i[0] for i in recommend_list_with_score]
        if len(recommend_list) > 1:
            sum_diversity_u += 1 - (s_similar.loc[recommend_list, recommend_list].sum().sum() - len(recommend_list)) / (0.5 * len(recommend_list) * (len(recommend_list) - 1))
            for i in recommend_list:
                ru_set.add(i)
        else:
            user_minus += 1
        print('\r%s' % u, end='', flush=True)
    coverage = len(ru_set) / len(s_rate.index)
    diversity = sum_diversity_u / (len(s_rate.columns) - user_minus)
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)
    # on whole dataset (9% support)
    # Coverage: 0.0113314447592068
    # Diversity: -0.43459952403435986
    # (8%)
    # Coverage: 0.018799896986865825
    # Diversity: -0.4167459430089829
