"""
使用基于物品的协同过滤进行推荐
"""

import numpy as np
import pandas as pd


def s_rate_mean(s_rate):
    """
    平均化矩阵内各用户评分

    :param s_rate: 用户评分矩阵：pandas.DataFrame
    :return: 平均化后的用户评分矩阵：pandas.DataFrame
    """
    return s_rate - s_rate.mean()


def matrix_prepare(s_rate, s_similar):
    """
    为协同过滤进行矩阵准备

    :param s_rate:    评分表：pandas.DataFrame
    :param s_similar: 各项目之间的相似性：pandas.DataFrame
    :return: 预测的评分表：pandas.DataFrame
    """
    s_predict = pd.DataFrame(index=s_rate.index, columns=s_rate.columns, dtype=np.float)
    for u in s_rate.columns:
        for i in s_rate.index:
            if np.isnan(s_rate.loc[i, u]):
                sij = s_similar[~s_rate[u].isnull()][i]
                # TODO: sij.sum==0时的情况，(60, 391)之后的值注意
                s_predict.loc[i, u] = (s_rate[~s_rate[u].isnull()][u] * sij).sum() / sij.sum()
            print('\r%s\t%s' % (u, i), end='', flush=True)
    return s_predict


def recommend_ibcf(s_predict, u, num=10):
    """
    生成推荐列表

    :param s_predict: 预测的评分表
    :param u: 用户ID
    :param num: 推荐列表中列表项数量
    :return:
    """
    return [
        (k, s_predict.loc[k, u])
        for k in s_predict[u].sort_values(ascending=False).index[:num]
    ]


def recommend_ibcf_single(s_rate, s_similar, u, num=10):
    s_rr = pd.DataFrame(index=s_rate.index, columns=['Score'], dtype=np.float)
    for i in s_rate.index:
        if np.isnan(s_rate.loc[i, u]):
            sij = s_similar[~s_rate[u].isnull()][i]
            s_rr.loc[i, 'Score'] = (s_rate[~s_rate[u].isnull()][u] * sij).sum() / sij.sum()
    rec_movies = dict([
        (k, s_rr.loc[k, 'Score'])
        for k in s_rr.sort_values(by='Score', ascending=False).index[:10]
    ])
    return rec_movies


if __name__ == '__main__':
    s_rate = pd.read_csv('../temp/s_rate.csv')
    s_rate = s_rate.set_index('MovieID')
    s_rate.rename(columns=int, inplace=True)

    s_similar = pd.read_csv('../temp/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)

    s_rate_mean = s_rate_mean(s_rate)

    s_predict = matrix_prepare(s_rate_mean, s_similar)

    print(recommend_ibcf(s_predict, 1, num=10))

    # RMSE
    # rmse = np.sqrt(
    #     ((s_rate_mean.fillna(0) - s_rate_predict[
    #         ~s_rate_mean.isnull()]) ** 2).sum().sum() / (
    #         s_rate_mean.count().count()))
    # print('RMSE: %s' % rmse)

    # 覆盖率 Coverage & 多样性 Diversity
    ru_set = set()
    sum_diversity_u = 0
    for u in s_rate.columns:
        recommend_list_with_score = recommend_ibcf(s_rate_mean, u)
        recommend_list = [i[0] for i in recommend_list_with_score]
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
    #
