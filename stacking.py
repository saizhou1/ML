# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :code
# @File     :stacking
# @Date     :2021/4/29 13:24
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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression


train_data = pd.read_csv('AAC.txt', sep=',', header=None)
train_data = np.vstack((train_data.iloc[:2559, 1:].values, train_data.iloc[2559:5118, 1:].values))
test_data = pd.read_csv('testAAC.txt', sep=',', header=None)
# test_data = np.vstack((test_data.iloc[:640, 1:].values, test_data.iloc[640:1280, 1:].values))
test_data = test_data.iloc[:, 1:].values
train_label = np.array([1 for i in range(2559)] + [0 for i in range(2559)])
test_label = np.array([1 for i in range(640)] + [0 for i in range(17264)])
x = np.vstack((train_data, test_data))
y = np.hstack((train_label, test_label))



clf1 = classifier = svm.SVC(C=2, kernel='rbf', gamma=30, decision_function_shape='ovo',probability=True)

clf2 = RandomForestClassifier(n_estimators=400,
                                         max_depth=None,
                                         max_features='auto',
                                         n_jobs=3,
                                         random_state=0,
                                       bootstrap=False
                                         )
clf3 = XGBClassifier(
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
 seed=27)
#
# clf4= AdaBoostClassifier(n_estimators=500, learning_rate=1.5, algorithm='SAMME.R', random_state=None)


lr = LogisticRegression(C=0.01)

estimators = [('SVM', clf1), ('Random Forest', clf2),('xgboost', clf3)]
sclf = StackingClassifier(estimators=estimators, final_estimator=lr)


sclf.fit(train_data,train_label)
tra_label = sclf.predict(train_data)  # 训练集的预测标签
tes_label = sclf.predict(test_data)  # 测试集的预测标签
print("训练集：", accuracy_score(train_label, tra_label))
print("测试集：", accuracy_score(test_label, tes_label))

matrix=confusion_matrix(test_label, tes_label,labels=[0,1])
TP=matrix[1,1]
TN=matrix[0,0]
FP=matrix[0,1]
FN=matrix[1,0]
sn=TP/(TP+FN)
sp=TN/(TN+FP)

decision_score = sclf.predict_proba(test_data)
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