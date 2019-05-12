'''
使用Word2Vec对用户行为进行学习，进而推荐
'''

import pandas as pd
import numpy as np
import gensim
import pickle

# 加载数据集
movies = pd.read_csv('./dataset/ml-out/movies.csv')
users = pd.read_csv('./dataset/ml-out/users.csv')
ratings = pd.read_csv('./dataset/ml-out/ratings.csv')
tags = pd.read_csv('./dataset/ml-out/tags.csv')
tag_scores = pd.read_csv('./dataset/ml-out/tag_scores.csv')

# 受皮尔逊距离影响，评分减去平均值
for i in users['UserID']:
    ratings.loc[ratings['UserID'] == i, 'Corr_rating'] = ratings.loc[
        ratings['UserID'] == i, 'Rating'] - ratings.loc[ratings['UserID'] ==
                                                        i, 'Rating'].mean()

# 固定排序
ratings_time = ratings.sort_values(by=['UserID', 'Timestamp'])
ratings_time.index = [i for i in range(ratings_time.shape[0])]

# 选取大于等于平均值的评分作为正样本，其他的为负样本
# 先找出来评分高的项目，再向前推，直到下一个项目是评分高的项目，或超出预定时间
# 5天内再次消费，则视为同时
# 30天内再次消费，则视为有先后关系
time_interval_min = 86400 * 5
time_interval_max = 86400 * 30
i = 1
# 按时间排序的特定用户的评分记录
# ratings_time[ratings_time['UserID'] == i]
# 按时间排序的特定用户的正评分记录
# ratings_time[(ratings_time['UserID'] == i) &
#              (ratings_time['Corr_rating'] >= 0)]
sel = []
for i in users['UserID']:
    for rating_pos_index in ratings_time[(ratings_time['UserID'] == i) & (
            ratings_time['Corr_rating'] >= 0)].index:
        sel_index = [str(int(ratings_time.loc[rating_pos_index]['MovieID']))]
        j = 1
        while True:
            front_data = ratings_time[(ratings_time['UserID'] == i)].shift(j)
            if np.isnan(front_data.loc[rating_pos_index, 'UserID']):
                # print('Invalid')
                break
            elif front_data.loc[rating_pos_index, 'Corr_rating'] > 0:
                # print('finish')
                break
            elif ratings_time.loc[
                    rating_pos_index, 'Timestamp'] - front_data.loc[
                        rating_pos_index, 'Timestamp'] > time_interval_max:
                # print('time exceed')
                break
            sel_index.append(
                str(int(front_data.loc[rating_pos_index]['MovieID'])))
            j += 1
        sel.append(list(reversed(sel_index)))
        print('\r%s\t%s' % (i, list(reversed(sel_index))), end='', flush=True)
print(' ')
print(sel)

with open('./temp/w2v_ds_list.pickle', 'wb') as f:
    pickle.dump(sel, f)

model = gensim.models.Word2Vec(sel, min_count=5, workers=10)


def recommend_wv(uid, model, num=10):
    mid = ratings[(ratings['UserID'] == uid) &
                  (ratings['Timestamp'] ==
                   ratings.loc[ratings['UserID'] == uid, 'Timestamp'].max()
                   )]['MovieID'].values[-1]
    return dict([(int(i[0]), i[1])
                 for i in model.wv.similar_by_word(str(mid), num)])


model.predict_output_word(['1801', '165', '428', '1690', '3257'])

model.wv.save_word2vec_format('./temp/w2cm.model')

if __name__ == '__main__':
    print(recommend_wv(4, model, num=10))
    print(model.predict_output_word(['1801', '165', '428', '1690', '3257']))
