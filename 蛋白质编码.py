# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :code
# @File     :蛋白质编码
# @Date     :2021/5/4 21:34
# @Author   :sai
# @Software :PyCharm
-------------------------------------------------
"""

import numpy as np
import os
from scipy.sparse import identity
#print(os.getcwd())#显示当前路径
#path="C:/Users/ASUS/Desktop/paper/pp_data"#数据处理路径
# os.chdir(path)
 # s=input("输入文件目录：")
f1 = open('onehot.txt','r')
All = f1.read().splitlines()

ee=[]
dd=[]
str1='OARNDCQEGHILKMFPSTWYVU'
for item in All:
    aa=np.zeros((41,22))
    for i in range(len(item)):
        b = str1.index(item[i])
        aa[i][b]=1
    a = aa.flatten()
    new_a = np.append(a,1)
    ee.append(new_a)
one_code = np.array(ee)