# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :fasta
# @File     :DT
# @Date     :2021/4/23 21:30
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

train_data = pd.read_csv('AAC.txt', sep=',', header=None)
train_data = np.vstack((train_data.iloc[:2559, 1:].values, train_data.iloc[2559:5118, 1:].values))
test_data = pd.read_csv('testAAC.txt', sep=',', header=None)
# test_data = np.vstack((test_data.iloc[:640, 1:].values, test_data.iloc[640:1280, 1:].values))
test_data = test_data.iloc[:, 1:].values
train_label = np.array([1 for i in range(2559)] + [0 for i in range(2559)])
test_label = np.array([1 for i in range(640)] + [0 for i in range(17264)])
x = np.vstack((train_data, test_data))
y = np.hstack((train_label, test_label))


from sklearn.tree import DecisionTreeRegressor
classifier = DecisionTreeRegressor(random_state=1)
classifier.fit(train_data, train_label.ravel())

# 也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

tra_label = classifier.predict(train_data).astype(int)  # 训练集的预测标签
tes_label = classifier.predict(test_data).astype(int)  # 测试集的预测标签
print("训练集：", accuracy_score(train_label, tra_label))
print("测试集：", accuracy_score(test_label, tes_label))

matrix=confusion_matrix(train_label, tra_label,labels=[0,1])
TP=matrix[1,1]
TN=matrix[0,0]
FP=matrix[0,1]
FN=matrix[1,0]
sn=TP/(TP+FN)
sp=TN/(TN+FP)

decision_score =  classifier.predict_proba(test_data)
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