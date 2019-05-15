#!/usr/bin/python3
"""
随机推荐（对照组）

Author: DingJunyao
Date: 2019-05-14 21:08
"""

import pandas as pd
import random


def random_recommend(items, num=10):
    random.shuffle(items)
    return items[:num]

if __name__ == '__main__':
    s_rate = pd.read_csv('../temp_ori/s_rate.csv')
    s_rate = s_rate.set_index('MovieID')
    s_rate.rename(columns=int, inplace=True)

    s_similar = pd.read_csv('../temp_ori/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)

    items = list(s_similar.index)
    recommend_list = random_recommend(items)
    print(recommend_list)

    # 覆盖率 Coverage & 多样性 Diversity
    ru_set = set()
    sum_diversity_u = 0
    for u in s_rate.columns:
        recommend_list = random_recommend(items)
        sum_diversity_u += 1 - (s_similar.loc[
                                    recommend_list, recommend_list].sum().sum() - len(
            recommend_list)) / (0.5 * len(recommend_list) * (
                len(recommend_list) - 1))
        for i in recommend_list:
            ru_set.add(i)
    coverage = len(ru_set) / len(s_rate.index)
    diversity = sum_diversity_u / len(s_rate.columns)
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)
    # on 1/10 dataset
    # Coverage: 1.0
    # Diversity: -0.25220797453089927
    # on whole dataset
    # Coverage: 1.0
    # Diversity: -0.276871350741371