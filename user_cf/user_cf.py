#!/usr/bin/python3
"""


Author: DingJunyao
Date: 2019-05-15 13:45
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


def recent_items(s_rate, rate_time):
    recent_df = s_rate
    recent_df[~(rate_time.max() - rate_time <= 365 * 86400)] = np.nan
    recent_df = recent_df.mask(recent_df >= 0, 1)
    return recent_df


def u_similar_on_buy(recent_df, ua, ub):
    k = recent_df[ua].sum() * recent_df[ub].sum()
    if k != 0:
        w_ab = (recent_df[ua] == recent_df[ub]).sum() / np.sqrt(k)
        return w_ab
    else:
        return 0.0


def matrix_prepare(s_rate, rate_time):
    recent_df = recent_items(s_rate, rate_time)
    u_similar_matrix_on_buy = pd.DataFrame(columns=recent_df.index, index=recent_df.index, dtype=np.float)
    u_similar_matrix_on_buy.index.name = 'UserID'
    for ii, i in enumerate(u_similar_matrix_on_buy.index):
        u_similar_matrix_on_buy.loc[i, i] = 1
        for j in u_similar_matrix_on_buy.index[ii:]:
            u_similar_matrix_on_buy.loc[i, j] = u_similar_on_buy(recent_df, i, j)
            u_similar_matrix_on_buy.loc[j, i] = u_similar_matrix_on_buy.loc[i, j]
    return u_similar_matrix_on_buy


if __name__ == '__main__':
    s_rate = pd.read_csv('../temp/s_rate.csv')
    s_rate = s_rate.set_index('MovieID')
    s_rate.rename(columns=int, inplace=True)
    rate_time = pd.read_csv('../temp/rate_time.csv')
    rate_time = rate_time.set_index('MovieID')
    rate_time.rename(columns=int, inplace=True)
    u_similar_matrix_on_buy = matrix_prepare(s_rate, rate_time)
    print(u_similar_matrix_on_buy)