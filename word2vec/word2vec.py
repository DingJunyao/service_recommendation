"""
使用Word2Vec对用户行为进行学习，进而推荐

Author: DingJunyao
Date：2019-05-17 18:23
"""

import pandas as pd
import gensim
import pickle
import time


def recommend_wv_series(series, model, num=10):
    """
    通过给定序列和Word2Vec模型，返回特定项数的推荐物品

    :param series: 序列：列表，元素为电影ID，一般是最近评分的电影
    :param model:  训练过的Word2Vec模型
    :return: 推荐列表：每一项为元组，包括电影ID和关联度
             错误码（0：成功；1：传的列表是空值；2：没有可推荐的列表；
                     3：传回了列表，但项数不合要求）
    """
    err = 0
    if series:
        recommend_list = model.predict_output_word(series, num)
        if not recommend_list:
            recommend_list = []
            err = 2
        elif len(recommend_list) < num:
            err = 3
        else:
            recommend_list = [(int(i[0]), i[1]) for i in recommend_list]
    else:
        recommend_list = []
        err = 1
    return recommend_list, err


def recommend_wv(ratings, model, uid, num=10, time_limit=0.2):
    """
    给定评价、评价时间和模型，根据用户最近评价的电影，为其推荐若干关联电影
    是对recommend_wv_series的封装

    :param ratings:    评分表
    :param model:      训练过的Word2Vec模型
    :param uid:        用户ID
    :param num:        项数，默认为10
    :param time_limit: 确定最近的评价记录时，确定最近若干比例的电影，默认为0.2
    :return: （见recommend_wv_series）
    """
    time_limit_reverse = 1 - time_limit
    ratings_u = ratings[(ratings['UserID'] == uid)]
    recent_items = ratings_u[(
                ratings_u['Timestamp'] > ratings_u['Timestamp'].quantile(
            time_limit_reverse))]['MovieID']
    recent_items_list = [str(int(i)) for i in recent_items]
    return recommend_wv_series(recent_items_list, model, num)


if __name__ == '__main__':
    ML_DS_PATH = '../dataset/ml-out'
    MATRIX_PATH = '../temp'

    ratings_old = pd.read_csv(ML_DS_PATH + '/ratings_old.csv')

    s_rate_old = pd.read_csv(MATRIX_PATH + '/s_rate_old.csv')
    s_rate_old = s_rate_old.set_index('MovieID')
    s_rate_old.rename(columns=int, inplace=True)

    with open(MATRIX_PATH + '/cluster_old.pickle', 'rb') as f:
        cluster = pickle.load(f)
    cluster = [[str(j) for j in i] for i in cluster]

    min_count_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    model_time_list = []
    rec_time_list = []
    precision_list = []
    recall_list = []
    f_measure_list = []
    coverage_list = []
    diversity_list = []
    err_1_list = []
    err_2_list = []
    err_3_list = []

    for i in min_count_list:
        print('=' * 20 + 'min_count=%s' % i + '=' * 20)
        model_start = time.time()  # 打点计时
        model = gensim.models.Word2Vec(cluster, min_count=i, workers=10)
        # model = gensim.models.Word2Vec.load('../temp_ori/w2cm.model')
        model_end = time.time()  # 打点计时
        # recommend_list_example, err = recommend_wv(ratings_old, model, 5)
        # print(recommend_list_example, err)
        print('Modeling Time: %s' % (model_end - model_start))
        model_time_list.append(model_end - model_start)
        model.save(MATRIX_PATH + '/w2cm.model')

        # 准确率 Precision & 召回率 Recall & 覆盖率 Coverage & 多样性 Diversity
        s_rate_new = pd.read_csv(MATRIX_PATH + '/s_rate_new.csv')
        s_rate_new = s_rate_new.set_index('MovieID')
        s_rate_new.rename(columns=int, inplace=True)

        s_similar = pd.read_csv(MATRIX_PATH + '/s_similar.csv')
        s_similar = s_similar.set_index('MovieID')
        s_similar.rename(columns=int, inplace=True)

        r_and_s_sum = 0
        r_sum = 0
        s_sum = 0
        ru_set = set()
        sum_diversity_u = 0
        user_minus = 0
        rec_time_sum = 0  # 打点计时
        err_1 = 0
        err_2 = 0
        err_3 = 0
        for u in s_rate_old.columns:
            select_set = set(s_rate_new[~s_rate_new[u].isnull()][u].index)
            rec_start = time.time()  # 打点计时
            recommend_list_with_score, err = recommend_wv(ratings_old, model, u)
            rec_end = time.time()  # 打点计时
            if err == 1:    # 空值
                err_1 += 1
            elif err == 2:
                err_2 += 1
            elif err == 3:
                err_3 += 1
            recommend_list = [i[0] for i in recommend_list_with_score]
            recommend_set = set(recommend_list)
            r_and_s_sum += len(recommend_set & select_set)
            r_sum += len(recommend_set)
            s_sum += len(select_set)
            if len(recommend_list) > 1:
                sum_diversity_u += 1 - (
                        s_similar.loc[
                            recommend_list, recommend_list
                        ].sum().sum() - len(recommend_list)) / (
                                           0.5 * len(recommend_list) * (
                                           len(recommend_list) - 1))
            else:
                user_minus += 1
            for i in recommend_list:
                ru_set.add(i)
            rec_time_sum += rec_end - rec_start  # 打点计时
            print('\r%s' % u, end='', flush=True)
        coverage = len(ru_set) / len(s_rate_old.index)
        diversity = sum_diversity_u / (len(s_rate_old.columns) - user_minus)
        rec_time = rec_time_sum / len(s_rate_old.columns)  # 打点计时
        precision = r_and_s_sum / r_sum
        recall = r_and_s_sum / s_sum
        f_measure = (2 * precision * recall) / (precision + recall)

        rec_time_list.append(rec_time)
        precision_list.append(precision)
        recall_list.append(recall)
        f_measure_list.append(f_measure)
        coverage_list.append(coverage)
        diversity_list.append(diversity)
        err_1_list.append(err_1)
        err_2_list.append(err_2)
        err_3_list.append(err_3)

        print('Average Recommend Time: %s' % rec_time)  # 打点计时
        print('Precision: %s' % precision)
        print('Recall: %s' % recall)
        print('F-Measure: %s' % f_measure)
        print('Coverage: %s' % coverage)
        print('Diversity: %s' % diversity)
        print('Blank List: %s' % err_1)
        print('No recommend: %s' % err_2)
        print('Recommend List not enough: %s' % err_3)

# # 覆盖率 Coverage & 多样性 Diversity
# s_rate_new = pd.read_csv(MATRIX_PATH + '/s_rate_new.csv')
# s_rate_new = s_rate_new.set_index('MovieID')
# s_rate_new.rename(columns=int, inplace=True)
#
# s_similar = pd.read_csv(MATRIX_PATH + '/s_similar.csv')
# s_similar = s_similar.set_index('MovieID')
# s_similar.rename(columns=int, inplace=True)
#
# ru_set = set()
# user_minus = 0
# sum_diversity_u = 0
# rec_time_sum = 0
# for u in s_rate.columns:
#     rec_single_start = time.time()  # 打点计时
#     recommend_list_with_score = recommend_wv(ratings, u, model)
#     rec_single_end = time.time()  # 打点计时
#     recommend_list = [i[0] for i in recommend_list_with_score]
#     if recommend_list:
#         sum_diversity_u += 1 - (s_similar.loc[
#                                     recommend_list, recommend_list
#                                 ].sum().sum() - len(
#             recommend_list)) / (0.5 * len(recommend_list) * (
#                 len(recommend_list) - 1))
#         for i in recommend_list:
#             ru_set.add(i)
#     else:
#         user_minus += 1
#     rec_time_sum += rec_single_end - rec_single_start
#     print('\r%s' % u, end='', flush=True)
# coverage = len(ru_set) / len(s_rate.index)
# diversity = sum_diversity_u / (len(s_rate.columns) - user_minus)
# rec_time = rec_time_sum / len(s_rate.columns)
# print('Coverage: %s' % coverage)
# print('Diversity: %s' % diversity)
# print('Recommend Average Time: %s' % rec_time)
