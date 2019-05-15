#!/usr/bin/python3
"""
仅依靠热门内容进行推荐（对照组）

Author: DingJunyao
Date: 2019-05-12 22:17
"""

import numpy as np
import pandas as pd


def matrix_prepare(s_rate):
    s_rate_hot = (s_rate.mean(axis=1) * s_rate.count(axis=1)).sort_values(
        ascending=False)
    return s_rate_hot


def recommend_hot(s_rate_hot, num=10):
    return [(i, s_rate_hot[i]) for i in s_rate_hot.index[:num]]


if __name__ == '__main__':
    s_rate = pd.read_csv('../temp/s_rate.csv')
    s_rate = s_rate.set_index('MovieID')
    s_rate.rename(columns=int, inplace=True)

    s_rate_hot = matrix_prepare(s_rate)
    recommend_list_with_score = recommend_hot(s_rate_hot)
    recommend_list = [i[0] for i in recommend_list_with_score]
    print(recommend_hot(s_rate_hot))

    # 覆盖率 Coverage
    ru_set = set()
    for i in recommend_list:
        ru_set.add(i)
    coverage = len(ru_set) / len(s_rate.index)
    print('Coverage: %s' % coverage)

    # 多样性 Diversity
    s_similar = pd.read_csv('../temp/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)
    diversity = 1 - (
                s_similar.loc[recommend_list, recommend_list].sum().sum() - len(
                    recommend_list)) / (0.5 * len(recommend_list) * (
                        len(recommend_list) - 1))
    print('Diversity: %s' % diversity)
    # on 1/10 dataset
    # Coverage: 0.02577319587628866
    # Diversity: -0.4066288582584414
    # on whole dataset
    # Coverage: 0.0025753283543651817
    # Diversity: -0.5584657705279685
