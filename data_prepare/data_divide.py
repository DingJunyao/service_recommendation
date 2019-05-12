import numpy as np
import pandas as pd
import random
import os
os.chdir('D:/service_recommendation/data_prepare')


s_rate = pd.read_csv('../temp/s_rate.csv')
s_rate = s_rate.set_index('MovieID')
s_rate.rename(columns=int, inplace=True)

time_effect_rate = pd.read_csv('../temp/time_effect_rate.csv')
time_effect_rate = time_effect_rate.set_index('MovieID')
time_effect_rate.rename(columns=int, inplace=True)

u_similar = pd.read_csv('../temp/u_similar.csv')
u_similar = u_similar.set_index('UserID')
u_similar.rename(columns=int, inplace=True)

users = list(u_similar.index)
random.shuffle(users)
user_len = int(len(users) / 5)
user_div = []
for i in range(5):
    user_div.append(users[user_len * i:user_len * (i + 1)])

for i in range(5):
    s_rate_test = s_rate[user_div[i]]
    s_rate_train = s_rate[list(set(users) ^ set(user_div[i]))]
    s_rate_test.to_pickle('../temp/divide/s_rate_%s_test.pickle' % i)
    s_rate_train.to_pickle('../temp/divide/s_rate_%s_train.pickle' % i)

    time_effect_rate_test = time_effect_rate[user_div[i]]
    time_effect_rate_train = time_effect_rate[list(
        set(users) ^ set(user_div[i]))]
    time_effect_rate_test.to_pickle(
        '../temp/divide/time_effect_rate_%s_test.pickle' % i)
    time_effect_rate_train.to_pickle(
        '../temp/divide/time_effect_rate_%s_train.pickle' % i)

    u_similar_test = u_similar.loc[user_div[i], user_div[i]]
    u_similar_train = u_similar.loc[list(set(users) ^ set(user_div[i])),
                                    list(set(users) ^ set(user_div[i]))]
    u_similar_test.to_pickle('../temp/divide/u_similar_%s_test.pickle' % i)
    u_similar_train.to_pickle('../temp/divide/u_similar_%s_train.pickle' % i)
