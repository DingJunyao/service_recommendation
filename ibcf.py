'''
使用基于物品的协同过滤进行推荐
'''

import numpy as np
import pandas as pd

s_rate = pd.read_csv('./temp/s_rate.csv')
s_rate = s_rate.set_index('MovieID')
s_rate.rename(columns=int, inplace=True)

s_similar = pd.read_csv('./temp/s_similar.csv')
s_similar = s_similar.set_index('MovieID')
s_similar.rename(columns=int, inplace=True)


def recommend_ibcf(u, num=10):
    s_rr = pd.DataFrame(index=s_rate.index, columns=['Score'], dtype=np.float)
    for i in s_rate.index:
        if np.isnan(s_rate.loc[i, u]):
            sij = s_similar[~s_rate[u].isnull()][i]
            s_rr.loc[i, 'Score'] = (s_rate[~s_rate[u].isnull()][u] *
                                    sij).sum() / sij.sum()
    rec_movies = dict([
        (k, s_rr.loc[k, 'Score'])
        for k in s_rr.sort_values(by='Score', ascending=False).index[:10]
    ])
    return rec_movies


if __name__ == '__main__':
    print(recommend_ibcf(4, num=10))
