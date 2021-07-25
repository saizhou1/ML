# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :fasta
# @File     :RF0
# @Date     :2021/3/28 22:27
# @Author   :sai
# @Software :PyCharm
-------------------------------------------------
"""
import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

from collections import Counter



train_data = pd.read_csv('AAC.txt', sep=',', header=None)
train_data = np.vstack((train_data.iloc[:2559, 1:].values, train_data.iloc[2559:7559, 1:].values))
train_label = np.array([1 for i in range(2559)] + [0 for i in range(5000)])
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(train_data, train_label )

train_data = X_smo
test_data = pd.read_csv('testAAC.txt', sep=',', header=None)
# test_data = np.vstack((test_data.iloc[:640, 1:].values, test_data.iloc[640:1280, 1:].values))
test_data = test_data.iloc[:1280, 1:].values
train_label = np.array([1 for i in range(2559)] + [0 for i in range(5000)]+[1 for i in range(2441)])
test_label = np.array([1 for i in range(640)] + [0 for i in range(640)])
x = np.vstack((train_data, test_data))
y = np.hstack((train_label, test_label))


from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# #一组十折交叉验证可视化
# rfc = RandomForestClassifier(n_estimators=196,random_state=90)
# score_pre = cross_val_score(rfc,train_data,train_label,cv=10)
# plt.plot(range(1,11),score_pre,label = "Random Forest")
# plt.legend()
# plt.show()

#调n_estimators，画学习曲线
# superpa = []
# for i in range(0,200,10):
#     rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1)
#     rfc_s = cross_val_score(rfc,train_data,train_label,cv=10).mean()
#     superpa.append(rfc_s)
# print(max(superpa),superpa.index(max(superpa)))
# plt.figure(figsize=[20,5])
# plt.plot(range(1,21),superpa)
# plt.show()

# #n_estimators=196、十折验证集的平均acc：0.8854998624021526
# scorel = []
# for i in range(190,210):
#     rfc = RandomForestClassifier(n_estimators=i,
#                                  n_jobs=-1,
#                                  random_state=90)
#     score = cross_val_score(rfc,train_data,train_label,cv=10).mean()
#     scorel.append(score)
# print(max(scorel),([*range(190,210)][scorel.index(max(scorel))]))
# plt.figure(figsize=[20,5])
# plt.plot(range(190,210),scorel)
# plt.show()

# #max_depth=21达到最优 0.8880419826320939
# param_grid = {'max_depth':np.arange(1, 51, 10)}
# # 一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
# # 但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够
# #   更应该画出学习曲线，来观察深度对模型的影响
# rfc = RandomForestClassifier(n_estimators=196
#                              ,random_state=90
#                            )
# GS = GridSearchCV(rfc,param_grid,cv=10)
# GS.fit(train_data,train_label)
# GS.best_params_
# GS.best_score_

#调整min_samples_leaf
param_grid={'min_samples_leaf':np.arange(1, 200, 50)}
#对于min_samples_split和min_samples_leaf,一般是从他们的最小值开始向上增加10或20
#面对高维度高样本量数据，如果不放心，也可以直接+50，对于大型数据，可能需要200~300的范围
#如果调整的时候发现准确率无论如何都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度
rfc = RandomForestClassifier(n_estimators=196
                             ,random_state=90
                             ,max_depth=21
                            ,criterion='gini'
                           )
GS = GridSearchCV(rfc,param_grid,cv=10)
GS.fit(train_data,train_label)
GS.best_params_
GS.best_score_


# #调整Criterion
# param_grid = {'criterion':['gini', 'entropy']}
# rfc = RandomForestClassifier(n_estimators=196
#                              ,random_state=90
#                              ,max_depth=21
#                            )
# GS = GridSearchCV(rfc,param_grid,cv=10)
# GS.fit(train_data,train_label)
# GS.best_params_
# GS.best_score_

# #调整max_features
# param_grid = {'max_features':np.arange(4,20,2)}
# """
# max_features是唯一一个即能够将模型往左（低方差高偏差）推，也能够将模型往右（高方差低偏差）推的参数。我
# 们需要根据调参前，模型所在的位置（在泛化误差最低点的左边还是右边）来决定我们要将max_features往哪边调。
# 现在模型位于图像左侧，我们需要的是更高的复杂度，因此我们应该把max_features往更大的方向调整，可用的特征
# 越多，模型才会越复杂。max_features的默认最小值是sqrt(n_features)，因此我们使用这个值作为调参范围的
# 最小值。
# """
# rfc = RandomForestClassifier(n_estimators=196
#                              ,random_state=90
#                              , max_depth=21
#                            )
# GS = GridSearchCV(rfc,param_grid,cv=10)
# GS.fit(train_data,train_label)
# GS.best_params_
# GS.best_score_


# 导入算法
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestRegressor
#
#
#
# # 建立树的个数
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # 最大特征的选择方式
# max_features = ['auto', 'sqrt']
# # 树的最大深度
# max_depth = [int(x) for x in np.linspace(10, 20, num = 2)]
# max_depth.append(None)#默认不限制最大深度
# # 样本采样方法
# bootstrap = [True, False]
#
# # Random grid
# random_grid = {'bootstrap': bootstrap}
# # 'n_estimators': n_estimators,
# #                'max_features': max_features,
# #                'max_depth': max_depth,
# #                'bootstrap': bootstrap}
# # 随机选择最合适的参数组合
# rf = RandomForestClassifier()
#
# rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
#                               n_iter = 100,scoring ='accuracy',
#                               cv = 10, verbose=2, random_state=42, n_jobs=-1)
# # scoring = 'neg_mean_absolute_error',
# # 执行寻找操作
# rf_random.fit(train_data, train_label)
# rf_random.best_params_
# rf_random.best_score_
# rf_random.cv_results_
# from sklearn.ensemble import RandomForestRegressor


#独立测试#最优为n_estimators=400，max_depth=
classifier= RandomForestClassifier(n_estimators=196
                             ,random_state=90
                             ,max_depth=21

                                         )

classifier.fit(train_data, train_label)
tra_label = classifier.predict(train_data)  # 训练集的预测标签
tes_label = classifier.predict(test_data)  # 测试集的预测标签
print("训练集：", accuracy_score(train_label, tra_label))
print("测试集：", accuracy_score(test_label, tes_label))

matrix=confusion_matrix(test_label, tes_label,labels=[0,1])
TP=matrix[1,1]
TN=matrix[0,0]
FP=matrix[0,1]
FN=matrix[1,0]
sn=TP/(TP+FN)
sp=TN/(TN+FP)

decision_score = classifier.predict_proba(test_data)
fprs, tprs, thresholds = roc_curve(test_label, decision_score[:, 1])

# plt.plot(fprs, tprs)
# plt.show()
roc_auc = auc(fprs, tprs)
plt.figure()
lw = 2
plt.plot(fprs, tprs, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.show()
