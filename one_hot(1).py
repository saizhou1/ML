# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:36:55 2021

@author: Kevan
"""
###输入s为文件路径
###输出为onehot编码矩阵
# def one_hot(s,path):
import numpy as np
import os
from scipy.sparse import identity
#print(os.getcwd())#显示当前路径
#path="C:/Users/ASUS/Desktop/paper/pp_data"#数据处理路径
# os.chdir(path)
 # s=input("输入文件目录：")
f1 = open('onehot.txt','r')
All = f1.read().splitlines()

h = len(All)
l = len(All[0])
p=All[0]

A = identity(21).toarray()  #21维单位矩阵
B = 'ACDEFGHIKLMNPQRSTVWYO'
B = list(B)
index = {}
for m in range(len(A)):
    index[B[m]] = A[m]
matrix_code1 = []
matrix_code2 = []
one_code1 = []
for i in range(h):
    matrix_code2 = []
    for j in range(l):
        for k in range(len(B)):
            if All[i][j] == B[k]:
                matrix_code1 = index[All[i][j]]
        matrix_code2.extend(matrix_code1)
    one_code1.append(matrix_code2)
one_code = np.array(one_code1)

#转化成二维形式，这个形式可以作为卷积神经网络的输入
train_dataset = one_code.reshape([-1,41,21,1])
# return X
#return one_code
   
    
        