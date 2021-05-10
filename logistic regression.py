# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :fasta
# @File     :logistic regression
# @Date     :2021/4/22 23:21
# @Author   :sai
# @Software :PyCharm
-------------------------------------------------
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# /# 读取数据
train_data = pd.read_csv('AAC.txt', sep=',', header=None)
train_data=np.vstack((train_data.iloc[:2559, 1:].values,train_data.iloc[2559:5118, 1:].values))
test_data = pd.read_csv('testAAC.txt', sep=',', header=None)
test_data=np.vstack((test_data.iloc[:640, 1:].values,test_data.iloc[640:1280, 1:].values))
train_label= np.array([1 for i in range(2559)]+[0 for i in range(2559)])
test_label= np.array([1 for i in range(640)]+[0 for i in range(640)])


lr = LogisticRegression(penalty='l2',  dual=False, tol=0.0001, C=1.0,fit_intercept=True, intercept_scaling=1,\
class_weight=None, random_state=None,\
solver='lbfgs', max_iter=100, multi_class='auto',\
verbose=0, warm_start=False, n_jobs=None,l1_ratio=None)
lr.fit(train_data,train_label)
lr.predict(test_data)
print("预测结果：",lr.score(test_data,test_label))



from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

tra_label = lr.predict(train_data)  # 训练集的预测标签
tes_label = lr.predict(test_data)  # 测试集的预测标签
print("训练集：", accuracy_score(train_label, tra_label))
print("测试集：", accuracy_score(test_label, tes_label))
matrix=confusion_matrix(train_label, tra_label,labels=[0,1])
TP=matrix[1,1]
TN=matrix[0,0]
FP=matrix[0,1]
FN=matrix[1,0]
sn=TP/(TP+FN)
sp=TN/(TN+FP)
precision=TP/(TP+FP)
F1=2*precision*sn/(precision+sn)

decision_score = lr.predict_proba(test_data)
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

