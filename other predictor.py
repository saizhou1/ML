# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :fasta
# @File     :other predictor
# @Date     :2021/4/12 21:15
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
fp = pd.read_csv('test.txt', sep='\n', header=None)
fp_list = list(fp.values)
n = len(fp_list)
k=1
for i in range(n):
    if (i + 1) % 2:
        xx.append(">Example"+str(k))
        k=k+1
        # a[2:7:2]从索引2开始到索引7停止,间隔为2。左闭右开。相同用法有arange(10)从0开始
    else:
        xx.append(fp_list[i][0])
T = open('iMethyl-PseAAC.fasta', 'w+')
for line in xx:
    T.write(line + '\n')
T.close()