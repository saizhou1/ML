# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :fasta
# @File     :SVM分类算法
# @Date     :2020/12/21 18:58
# @Author   :sai
# @Software :PyCharm
-------------------------------------------------
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np
import requests
import pandas as pd
import os
import json
import random
#将AAC和binary特征合并
# xx = []
# fp = pd.read_csv('AAC.txt', sep='\n', header=None)
# fp_list = list(fp.values)
# n = len(fp_list)
# fp1 = pd.read_csv('binary.txt', sep='\n', header=None)
# fp1_list = list(fp1.values)
# n = len(fp1_list)
# for i in range(n):
#     xx.append(fp_list[i][0] + fp1_list[i][0][1:])
#
# T = open('feature.txt', 'w+')
# for line in xx:
#     T.write(line + '\n')

df = pd.read_csv('AAC.txt', sep=',', header=None)  # 读取数据
X = df.iloc[:, 1:].values
y = df.iloc[:,0].values

data = scale(X)  # 标准化，标准化之后就自动根据协方差矩阵进行主成分分析了
# data2 = np.corrcoef(np.transpose(data)) # 没有必要单独计算协方差阵或相关系数阵
pca = PCA()  # 可以调整主成分个数，n_components = 1
pca.fit(data)
fea = pca.explained_variance_  # 输出特征根
contri = pca.explained_variance_ratio_  # 输出解释方差比
compon = pca.components_  # 输出主成分
