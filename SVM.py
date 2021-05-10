# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :fasta
# @File     :SVM
# @Date     :2021/3/25 22:58
# @Author   :sai
# @Software :PyCharm
-------------------------------------------------
"""
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn


# # define converts(字典)
# def Iris_label(s):
#     it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
#     return it[s]


# 1.读取数据集
# path = 'Iris.data'
# data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: Iris_label})
# # converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)x
# print(data.shape)

# 2.划分数据与标签
# x, y = np.split(data, indices_or_sections=[4], axis=1)  # x为数据，y为标签

# df = pd.read_csv('AAC.txt', sep=',', header=None)  # 读取数据
# x = df.iloc[:, 1:].values
# y = df.iloc[:,0].values
# x = x[:, 0:2]
train_data = pd.read_csv('AAC.txt', sep=',', header=None)
train_data=np.vstack((train_data.iloc[:2559, 1:].values,train_data.iloc[2559:5118, 1:].values))
test_data = pd.read_csv('testAAC.txt', sep=',', header=None)
test_data=np.vstack((test_data.iloc[:640, 1:].values,test_data.iloc[640:1280, 1:].values))
train_label= np.array([1 for i in range(2559)]+[0 for i in range(2559)])
test_label= np.array([1 for i in range(640)]+[0 for i in range(640)])
x=np.vstack((train_data,test_data))
y=np.hstack((train_label,test_label))

# x = x[:, 15:17]
# train_data = train_data[:, 15:17]
# test_data = test_data[:, 15:17]
# train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.6,
#                                                                   test_size=0.4)  # sklearn.model_selection.
# print(train_data.shape)

# 3.训练svm分类器
classifier = svm.SVC(C=2, kernel='rbf', gamma=30, decision_function_shape='ovo',probability=True)  # ovr:一对多策略
classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先




# 4.计算svc分类器的准确率
# print("训练集：", classifier.score(train_data, train_label))
# print("测试集：", classifier.score(test_data, test_label))

# 也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

tra_label = classifier.predict(train_data)  # 训练集的预测标签
tes_label = classifier.predict(test_data)  # 测试集的预测标签
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

# 查看决策函数
print('train_decision_function:\n', classifier.decision_function(train_data))  # (90,3)
print('predict_result:\n', classifier.predict(train_data))


# # 5.绘制图形
# # 确定坐标轴范围
# x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
# x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
# x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
# #表示x1按列展开，x2按行展开，200j表示复数从起点到终点分成200等份，左闭右闭区间
# grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# # 指定默认字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# # 设置颜色
# cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0'])
# cm_dark = matplotlib.colors.ListedColormap(['g', 'r'])
#
# grid_hat = classifier.predict(grid_test)  # 预测分类值
# grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
#
# plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
# plt.scatter(x[:, 0], x[:, 1], c=y.T, s=30, cmap=cm_dark)  # 样本
# plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label.T, s=30, edgecolors='k', zorder=2,
#             cmap=cm_dark)  # 圈中测试集样本点
# plt.xlabel('花萼长度', fontsize=13)
# plt.ylabel('花萼宽度', fontsize=13)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('鸢尾花SVM二特征分类')
# plt.show()
