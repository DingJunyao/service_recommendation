"""
使用Word2Vec对用户行为进行学习，进而推荐
"""

import pandas as pd
import numpy as np
import gensim
import pickle


# 加载数据集
# movies = pd.read_csv('../dataset/ml-out/movies.csv')
# users = pd.read_csv('../dataset/ml-out/users.csv')

# tags = pd.read_csv('../dataset/ml-out/tags.csv')
# tag_scores = pd.read_csv('../dataset/ml-out/tag_scores.csv')

# 受皮尔逊距离影响，评分减去平均值
# for i in users['UserID']:
#     ratings.loc[ratings['UserID'] == i, 'Corr_rating'] = ratings.loc[
#         ratings['UserID'] == i, 'Rating'] - ratings.loc[ratings['UserID'] ==
#                                                         i, 'Rating'].mean()

# 固定排序
# ratings_time = ratings.sort_values(by=['UserID', 'Timestamp'])
# ratings_time.index = [i for i in range(ratings_time.shape[0])]

# 选取大于等于平均值的评分作为正样本，其他的为负样本
# 先找出来评分高的项目，再向前推，直到下一个项目是评分高的项目，或超出预定时间
# 5天内再次消费，则视为同时
# 30天内再次消费，则视为有先后关系
# time_interval_min = 86400 * 5
# time_interval_max = 86400 * 30
# i = 1
# 按时间排序的特定用户的评分记录
# ratings_time[ratings_time['UserID'] == i]
# 按时间排序的特定用户的正评分记录
# ratings_time[(ratings_time['UserID'] == i) &
#              (ratings_time['Corr_rating'] >= 0)]
# sel = []
# for i in users['UserID']:
#     for rating_pos_index in ratings_time[(ratings_time['UserID'] == i) & (
#             ratings_time['Corr_rating'] >= 0)].index:
#         sel_index = [str(int(ratings_time.loc[rating_pos_index]['MovieID']))]
#         j = 1
#         while True:
#             front_data = ratings_time[(ratings_time['UserID'] == i)].shift(j)
#             if np.isnan(front_data.loc[rating_pos_index, 'UserID']):
#                 # print('Invalid')
#                 break
#             elif front_data.loc[rating_pos_index, 'Corr_rating'] > 0:
#                 # print('finish')
#                 break
#             elif ratings_time.loc[
#                     rating_pos_index, 'Timestamp'] - front_data.loc[
#                         rating_pos_index, 'Timestamp'] > time_interval_max:
#                 # print('time exceed')
#                 break
#             sel_index.append(
#                 str(int(front_data.loc[rating_pos_index]['MovieID'])))
#             j += 1
#         sel.append(list(reversed(sel_index)))
#         print('\r%s\t%s' % (i, list(reversed(sel_index))), end='', flush=True)
# print(' ')
# print(sel)

# with open('./temp/w2v_ds_list.pickle', 'wb') as f:
#     pickle.dump(sel, f)


def recommend_wv(ratings, uid, model, num=10):
    ratings_u = ratings[(ratings['UserID'] == uid)]
    recent_items = ratings_u[
        (ratings_u['Timestamp'].max() - ratings_u['Timestamp'] <= 30 * 86400)][
        'MovieID']
    recent_items_list = list(recent_items.values)
    recent_items_list = [str(i) for i in recent_items_list]
    predict_list = model.predict_output_word(recent_items_list, num)
    if not predict_list:
        predict_list = []
    predict_list = [(int(i[0]), i[1]) for i in predict_list]
    return predict_list


# model.predict_output_word(['1801', '165', '428', '1690', '3257'])

# model.save('../temp/w2cm.model')

if __name__ == '__main__':
    ratings = pd.read_csv('../dataset/ml-out/ratings.csv')

    s_rate = pd.read_csv('../temp_ori/s_rate.csv')
    s_rate = s_rate.set_index('MovieID')
    s_rate.rename(columns=int, inplace=True)

    s_similar = pd.read_csv('../temp_ori/s_similar.csv')
    s_similar = s_similar.set_index('MovieID')
    s_similar.rename(columns=int, inplace=True)
    # with open('../temp_ori/cluster.pickle', 'rb') as f:
    #     sel = pickle.load(f)
    # sel = [[str(j) for j in i] for i in sel]
    # model = gensim.models.Word2Vec(sel, min_count=5, workers=10)
    model = gensim.models.Word2Vec.load('../temp/w2cm.model')
    print(recommend_wv(ratings, 4, model))

    # 覆盖率 Coverage & 多样性 Diversity
    ru_set = set()
    user_minus = 0
    sum_diversity_u = 0
    for u in s_rate.columns:
        recommend_list_with_score = recommend_wv(ratings, u, model)
        recommend_list = [i[0] for i in recommend_list_with_score]
        if recommend_list:
            sum_diversity_u += 1 - (s_similar.loc[
                                        recommend_list, recommend_list
                                    ].sum().sum() - len(
                recommend_list)) / (0.5 * len(recommend_list) * (
                    len(recommend_list) - 1))
            for i in recommend_list:
                ru_set.add(i)
        else:
            user_minus += 1
        print('\r%s' % u, end='', flush=True)
    coverage = len(ru_set) / len(s_rate.index)
    diversity = sum_diversity_u / (len(s_rate.columns) - user_minus)
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)

    # 覆盖率 Coverage & 多样性 Diversity & 准确度 Precision & 召回率 Recall
    ru_set = set()
    len_ru = 0
    len_tu = 0
    len_ru_and_tu = 0
    user_minus = 0
    sum_diversity_u = 0
    # TODO: 将用户最近选择的项目与以往的项目分开，使用以往的项目进行预测，检查与选择的契合度
    for u in s_rate.columns:
        recommend_list_with_score = recommend_wv(ratings, u, model)
        recommend_list = [i[0] for i in recommend_list_with_score]
        if recommend_list:
            sum_diversity_u += 1 - (s_similar.loc[
                                        recommend_list, recommend_list
                                    ].sum().sum() - len(
                recommend_list)) / (0.5 * len(recommend_list) * (
                    len(recommend_list) - 1))
            for i in recommend_list:
                ru_set.add(i)
        else:
            user_minus += 1
        print('\r%s' % u, end='', flush=True)
    coverage = len(ru_set) / len(s_rate.index)
    diversity = sum_diversity_u / (len(s_rate.columns) - user_minus)
    print('Coverage: %s' % coverage)
    print('Diversity: %s' % diversity)

    # on whole dataset
    # Coverage: 0.5454545454545454
    # Diversity: -0.30991427079154976
