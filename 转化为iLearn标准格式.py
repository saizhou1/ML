# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :fasta
# @File     :转化为iLearn标准格式
# @Date     :2020/12/30 12:22
# @Author   :sai
# @Software :PyCharm
-------------------------------------------------
"""
import requests
import pandas as pd
import os
import json
import random

print(os.getcwd())  # 显示当前路径
xx = []
fp = pd.read_csv('ptrain.txt', sep='\n', header=None)
fp_list = list(fp.values)
n = len(fp_list)
for i in range(640):
    if (i + 1) % 2:
        xx.append(fp_list[i][0][0]+fp_list[i][0][4:10]+"_"+fp_list[i][0][13:]+"|1|training")
        # a[2:7:2]从索引2开始到索引7停止,间隔为2。左闭右开。相同用法有arange(10)从0开始
    else:
        xx.append(fp_list[i][0])
T = open('xiuzheng.fasta', 'w+')
for line in xx:
    T.write(line + '\n')


yy = []
fp = pd.read_csv('ntrain.txt', sep='\n', header=None)
fp_list = list(fp.values)
n = len(fp_list)
for i in range(640):
    if (i + 1) % 2:
        yy.append(fp_list[i][0][0]+fp_list[i][0][4:10]+"_"+fp_list[i][0][13:]+"|0|training")
    else:
        yy.append(fp_list[i][0])
for line in yy:
# file.write(js)
    T.write(line + '\n')


zz = []
fp = pd.read_csv('test.txt', sep='\n', header=None)
fp_list = list(fp.values)
for i in range(150):
    if (i + 1) % 2:
        zz.append(fp_list[i][0][0]+fp_list[i][0][4:10]+"_"+fp_list[i][0][13:]+"|1|testing")
    else:
        zz.append(fp_list[i][0])
for i in range(641,791):
    if (i + 1) % 2:
        zz.append(fp_list[i][0][0]+fp_list[i][0][4:10]+"_"+fp_list[i][0][13:]+"|0|testing")
    else:
        zz.append(fp_list[i][0])
for line in zz:
# file.write(js)
    T.write(line + '\n')
T.close()

# 将ID输入文件
# js = json.dumps(Id)
# file = open('Id.txt', 'w')
# file.close()
# # 读取Id
# file = open('Id.txt', 'r')
# js = file.read()
# dic = json.loads(js)
# Id_site = list(dic)
# # file.close()
# 将ID提出来
# x = []
# fr = pd.read_csv('fasta_1(1).fasta', sep='\n', header=None)
# fr_list = list(fr.values)
# n = len(fr_list)
# for i in range(n):
#     if (i + 1) % 2:
#         x.append(fr_list[i])
# Id = []
# m = len(x)
# for i in range(m):
#     Id.append(x[i][0][4:10])


print(os.getcwd())  # 显示当前路径
xx = []
fp = pd.read_csv('ptrain.txt', sep='\n', header=None)
fp2 = pd.read_csv('ntrain.txt', sep='\n', header=None)

fp_list = list(fp.values)
fp2_list = list(fp2.values)
n = len(fp_list)
for i in range(5118):
    if (i ) % 2:
        xx.append(fp_list[i][0])
        # a[2:7:2]从索引2开始到索引7停止,间隔为2。左闭右开。相同用法有arange(10)从0开始

for j in range(5118):
    if (j) % 2:
        xx.append(fp2_list[i][0])

T = open('onehot1.txt', 'w+')
for line in xx:
    T.write(line + '\n')