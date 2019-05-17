#!/usr/bin/python3
"""
通过Mean Shift算法对购买记录进行聚类

Author: DingJunyao
Date: 2019-05-14 10:01
"""

import pandas as pd
from sklearn.cluster import MeanShift
import numpy as np
import time
import pickle


MATRIX_PATH = '../temp-sample'
rate_time = pd.read_csv(MATRIX_PATH + '/rate_time_old.csv')
rate_time = rate_time.set_index('MovieID')
rate_time.rename(columns=int, inplace=True)


cluster_start = time.time()  # 打点计时
cluster = []
for u in rate_time.columns:
    m_id = []
    m_series = []
    ratings_u = rate_time[u].dropna()
    if len(ratings_u) < 2:
        continue
    for line in ratings_u.iteritems():
        m_id.append(line[0])
        m_series.append(line[1] / 86400)
    m_series_np = np.array(m_series).reshape((-1, 1))
    clustering = MeanShift(bandwidth=25).fit(m_series_np)
    cluster_dict = {}
    for i in zip(m_id, clustering.labels_):
        if i[1] not in cluster_dict:
            cluster_dict[i[1]] = [i[0]]
        else:
            cluster_dict[i[1]].append(i[0])
    for i in cluster_dict.values():
        if len(i) > 1:
            cluster.append(i)
cluster_end = time.time()  # 打点计时
with open(MATRIX_PATH + '/cluster_old.pickle', 'wb') as f:
    pickle.dump(cluster, f)
print('Cluster Time: %s' % (cluster_end - cluster_start))  # 打点计时
