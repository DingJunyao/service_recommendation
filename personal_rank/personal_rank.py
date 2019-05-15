#!/usr/bin/python3
"""


Author: DingJunyao
Date: 2019-05-13 18:22
"""

import numpy as np
import pandas as pd


def s_rate_mean(s_rate):
    """
    平均化矩阵内各用户评分

    :param s_rate: 用户评分矩阵：pandas.DataFrame
    :return:       平均化后的用户评分矩阵：pandas.DataFrame
    """
    return s_rate - s_rate.mean()


# def personal_rank(graph, alpha, user, steps, epoch=0.0001):
#     rank = {}
#     for x in graph.keys():
#         rank[x] = 0
#     rank[user] = 1
#
#     for step in steps:
#         tmp = {}
#         for x in graph.keys:
#             tmp[x] = 0
#         for i, ri in graph.items():
#             for j in ri.keys():
#                 if j not in tmp:
#                     tmp[j] = 0
#                 tmp[j] += alpha * rank[i] / float(len(ri))
#                 if j == user:
#                     tmp[j] += 1 - alpha
#         check = []
#         for k in tmp.keys():
#             check.append(tmp[k] - rank[k])
#         if sum(check) <= epoch:
#             break
#         rank = tmp
#         if step % 20 == 0:
#             print('iter: %s' % step)
#     return rank


def graph_gen(s_rate, rate_limit=4):
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


def personal_rank(graph, vertex, u, num=10, alpha=0.8, bought=None):
    n = graph.shape[0]
    r_0 = np.zeros((n, 1))
    r_0[vertex.index('U_%s' % u)][0] = 1
    A = np.eye(n) - alpha * graph.T
    b = (1 - alpha) * r_0
    r = np.linalg.solve(A, b)
    scores = pd.DataFrame(r, index=vertex, columns=['Score'])
    scores = scores[scores.index.str.startswith(
        'S_')].sort_values(by='Score', ascending=False)['Score']
    scores.rename(index=lambda x: int(x[2:]), inplace=True)
    if bought is not None:
        scores.drop(bought, inplace=True)
    return [(i, scores.loc[i]) for i in scores.index[:num]]


def personal_rank_user(graph, vertex, u, num=10, alpha=0.8):
    return personal_rank(graph, vertex, u, num, alpha,
                         s_rate[u][~s_rate[u].isnull()].index)


if __name__ == '__main__':
    s_rate = pd.read_csv('../temp_ori/s_rate.csv')
    s_rate = s_rate.set_index('MovieID')
    s_rate.rename(columns=int, inplace=True)
    s_rate_mean = s_rate_mean(s_rate)
    g, v = graph_gen(s_rate_mean, 0)
    print(personal_rank_user(g, v, 4, 10, 0.8))

    s_similar = pd.read_csv('../temp_ori/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)

    # 覆盖率 Coverage & 多样性 Diversity
    ru_set = set()
    sum_diversity_u = 0
    for u in s_rate.columns:
        recommend_list_with_score = personal_rank_user(g, v, u)
        recommend_list = [i[0] for i in recommend_list_with_score]
        sum_diversity_u += 1 - (s_similar.loc[
                                    recommend_list, recommend_list].sum().sum() - len(
            recommend_list)) / (0.5 * len(recommend_list) * (
                len(recommend_list) - 1))
        for i in recommend_list:
            ru_set.add(i)
        print('\r%s' % (u), end='', flush=True)
    coverage = len(ru_set) / len(s_rate.index)
    diversity = sum_diversity_u / len(s_rate.columns)
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)
    # on 1/10 dataset
    # Coverage: 0.34536082474226804
    # Diversity: -0.2581819489140455
