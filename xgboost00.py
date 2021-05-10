# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :code
# @File     :xgboost
# @Date     :2021/4/29 13:48
# @Author   :sai
# @Software :PyCharm
-------------------------------------------------
"""

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot

train_data = pd.read_csv('AAC.txt', sep=',', header=None)
train_data = np.vstack((train_data.iloc[:2559, 1:].values, train_data.iloc[2559:5118, 1:].values))
test_data = pd.read_csv('testAAC.txt', sep=',', header=None)
# test_data = np.vstack((test_data.iloc[:640, 1:].values, test_data.iloc[640:1280, 1:].values))
test_data = test_data.iloc[:, 1:].values
train_label = np.array([1 for i in range(2559)] + [0 for i in range(2559)])
test_label = np.array([1 for i in range(640)] + [0 for i in range(17264)])
x = np.vstack((train_data, test_data))
y = np.hstack((train_label, test_label))

# fit model no training data
model = XGBClassifier()
model.fit(x, y)
# plot feature importance
plot_importance(model)
pyplot.show()


from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

# grid search
model = XGBClassifier()
learning_rate = [0.3,0.5,0.6,0.7,0.8,1]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(x, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
params = grid_result.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with: %r" % (mean, param))

xgb1 = XGBClassifier(
 learning_rate =0.6,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27,eval_metric='mlogloss')


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

classifier = XGBClassifier()
eval_set = [(test_data, test_label)]
classifier.fit(train_data, train_label, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
# make predictions for test data
y_pred = classifier.predict(test_data)
predictions = [round(value) for value in y_pred]
# evaluate predictions
classifier.fit(train_data, train_label)
tra_label = classifier.predict(train_data)  # 训练集的预测标签
tes_label = classifier.predict(test_data)  # 测试集的预测标签
print("训练集：", accuracy_score(train_label, tra_label))
print("测试集：", accuracy_score(test_label, tes_label))
