#!/usr/bin/python3
"""


Author: DingJunyao
Date: 2019-05-14 10:01
"""

import pandas as pd
from sklearn.cluster import MeanShift
import numpy as np
import time
import pickle


if __name__ == '__main__':
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '任务开始')
    ml_ds_path = '../dataset/ml-out-reduce'
    ratings = pd.read_csv(ml_ds_path + '/ratings.csv')
    users = pd.read_csv(ml_ds_path + '/users.csv')
    cluster = []
    for u in users['UserID']:
        m_id = []
        m_series = []
        ratings_u = ratings[ratings['UserID'] == u]
        if len(ratings_u) < 2:
            continue
        for line in ratings_u.itertuples():
            m_id.append(line[2])
            m_series.append(line[4] / 86400)
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
        print('\r%s' % u, end='', flush=True)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '聚类完成')

    with open('../temp/cluster.pickle', 'wb') as f:
        pickle.dump(cluster, f)
