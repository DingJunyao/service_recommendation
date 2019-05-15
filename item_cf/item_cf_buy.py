#!/usr/bin/python3
"""
基于共同喜换物品的用户列表计算

Author: DingJunyao
Date: 2019-05-15 16:05
"""

import pandas as pd
import numpy as np
import pickle
import itertools


def item_similar_on_buy(item_buy_number, s_bought, i, j):
    if item_buy_number[i] * item_buy_number[j] != 0:
        w_ij = s_bought.loc[i, j] / np.sqrt(item_buy_number[i] * item_buy_number[j])
    else:
        w_ij = 0.0
    return w_ij


def s_bought_gen(cluster, s_rate):
    s_bought = pd.DataFrame(columns=s_rate.index, index=s_rate.index,
                            dtype=np.int).fillna(0)
    for ii, i in enumerate(cluster):
        for j in itertools.combinations(i, 2):
            s_bought.loc[j[0], j[1]] += 1
            s_bought.loc[j[1], j[0]] += 1
        # print('\r%s' % ii, end='', flush=True)
    return s_bought


def s_similar_on_buy_gen(s_rate, s_bought):
    s_similar_on_buy = pd.DataFrame(columns=s_rate.index, index=s_rate.index,
                                    dtype=np.float)
    item_buy_number = s_rate.T.count()
    for ii, i in enumerate(s_similar_on_buy.index):
        s_similar_on_buy.loc[i, i] = 1
        for j in s_similar_on_buy.index[ii + 1:]:
            s_similar_on_buy.loc[i, j] = item_similar_on_buy(item_buy_number, s_bought, i, j)
            s_similar_on_buy.loc[j, i] = s_similar_on_buy.loc[i, j]
        # print('\r%s' % ii, end='', flush=True)
    return s_similar_on_buy


def recent_items_df_gen(s_rate, rate_time):
    recent_items_df = s_rate
    recent_items_df[~(rate_time.max() - rate_time <= 365 * 86400)] = np.nan
    recent_items_df = recent_items_df.mask(recent_items_df >= 0, 1)
    return recent_items_df


def recent_items_gen(recent_items_df, u):
    return list(recent_items_df[u][~recent_items_df[u].isnull()].index)


def recommend_icf_series(s_similar_on_buy, recent_items, num=10):
    recommend_dict = {}
    for i in recent_items:
        recommend_series = s_similar_on_buy.drop(i)[i].sort_values(ascending=False)
        recommend_series = list(zip(recommend_series.index, recommend_series))
        for j in recommend_series:
            if j[0] not in recommend_dict:
                recommend_dict[j[0]] = j[1]
            else:
                if j[1] > recommend_dict[j[0]]:
                    recommend_dict[j[0]] = j[1]
    recommend_list_out = list(recommend_dict.items())
    recommend_list_out.sort(key=lambda x: x[1], reverse=True)
    if recommend_list_out:
        return recommend_list_out[:num]
    else:
        return []


def recommend_icf(s_similar_on_buy, recent_items_df, u, num=10):
    recent_items = recent_items_gen(recent_items_df, u)
    if recent_items:
        return recommend_icf_series(s_similar_on_buy, recent_items, num)
    else:
        return []



if __name__ == '__main__':
    s_rate = pd.read_csv('../temp/s_rate.csv')
    s_rate = s_rate.set_index('MovieID')
    s_rate.rename(columns=int, inplace=True)
    rate_time = pd.read_csv('../temp/rate_time.csv')
    rate_time = rate_time.set_index('MovieID')
    rate_time.rename(columns=int, inplace=True)
    s_similar = pd.read_csv('../temp/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)
    # s_bought = pd.DataFrame(columns=s_rate.index, index=s_rate.index, dtype=np.int).fillna(0)
    # s_similar_on_buy = pd.DataFrame(columns=s_rate.index, index=s_rate.index, dtype=np.float)
    with open('../temp/cluster.pickle', 'rb') as f:
        cluster = pickle.load(f)
    s_bought = s_bought_gen(cluster, s_rate)
    # for ii, i in enumerate(cluster):
    #     for j in itertools.combinations(i, 2):
    #         s_bought.loc[j[0], j[1]] += 1
    #         s_bought.loc[j[1], j[0]] += 1
    #     print('\r%s' % ii, end='', flush=True)
    # item_buy_number = s_rate.T.count()
    # for ii, i in enumerate(s_similar_on_buy.index):
    #     s_similar_on_buy.loc[i, i] = 1
    #     for j in s_similar_on_buy.index[ii + 1:]:
    #         s_similar_on_buy.loc[i, j] = item_similar_on_buy(item_buy_number, s_bought, i, j)
    #         s_similar_on_buy.loc[j, i] = s_similar_on_buy.loc[i, j]
    #     print('\r%s' % ii, end='', flush=True)
    s_similar_on_buy = s_similar_on_buy_gen(s_rate, s_bought)
    recent_df = recent_items_df_gen(s_rate, rate_time)
    recommend_item = recommend_icf(s_similar_on_buy, recent_df, 15, num=10)
    print(recommend_item)

    # 覆盖率 Coverage & 多样性 Diversity
    ru_set = set()
    sum_diversity_u = 0
    user_minus = 0
    for u in s_rate.columns:
        recommend_list_with_score = recommend_icf(s_similar_on_buy, recent_df, u, num=10)
        recommend_list = [i[0] for i in recommend_list_with_score]
        if len(recommend_list) > 1:
            sum_diversity_u += 1 - (s_similar.loc[
                                        recommend_list, recommend_list].sum().sum() - len(
                recommend_list)) / (0.5 * len(recommend_list) * (
                        len(recommend_list) - 1))
            for i in recommend_list:
                ru_set.add(i)
        else:
            user_minus += 1
        print('\r%s' % u, end='', flush=True)
    coverage = len(ru_set) / len(s_rate.index)
    diversity = sum_diversity_u / len(s_rate.columns)
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)
    # on 1/10 dataset
    # Coverage: 0.5077319587628866
    # Diversity: -0.32148315065758376


