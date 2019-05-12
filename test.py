'''import xlearn as xl
import numpy as np
import pandas as pd
s_rate = pd.read_csv('./temp/s_rate.csv')
s_rate = s_rate.set_index('MovieID')
s_rate.rename(columns=int, inplace=True)
X_train = s_rate
y_train = s_rate.index.values

xdm_train = xl.DMatrix(X_train, y_train)

fm_model = xl.create_fm()
fm_model.setTrain(xdm_train)
param = {'task':'reg', 'lr':0.2, 'lambda':0.002, 'metric': 'rmse'}
fm_model.fit(param, "./model_fm.out")
fm_model.setTest(xdm_train)
res = fm_model.predict('./model_fm.out')
print(res) '''

import pickle
import pandas as pd
import numpy as np
import gensim
with open('./temp/w2v_ds_list.pickle', 'rb') as f:
    a = pickle.load(f)
model = gensim.models.Word2Vec(a, min_count=5, workers=10)
print(model.predict_output_word(['1509', '3046', '3418']))
