# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :毕业设计
# @File     :数据练习
# @Date     :2
#
#020/12/10 14:57
# @Author   :sai
# @Software :PyCharm
-------------------------------------------------
"""
# import re  # 正则表达式，进行文字匹配
# import urllib.error
# import urllib.request  # 指定URL，获取网页数据
#
# import bs4  # 网页解析，获取数据
# import xlwt  # 进行excel操作
# import sqlite3  # 进行sqlite数据库操作
# def main(a):
#     print("hello")
#
#
# if __name__ == "__main__":  # 当程序执行时
#     # 调用函数
#     main()

'''transform string into dictionary

s is input string 
d is dictionary to restore every bit in string
'''

import math

def histogram(s, old_d):     #计字母数目函数
    d = old_d
    for c in s:
        d[c] = d.get(c, 0) + 1
    return d

'''This function can calculate the frequency of every letter in alphabet

'''
fin = open("ntrain.txt")   #负样本训练集
new_list = []  # 新列表
sum1 = [0 for i in range(41)]
for line in fin:
    rs = line.rstrip('\n')  # delete the '\n' after every letter
    new_list.append(rs)  # new_list is used to restore letters
old_d = dict()  # initialize the dictionary
for i in range(0, 41): #0遍历到第40个位置
    old_d[i] = dict();
for i in range(len(new_list)):  # calculate the leter
    for j in range(41):
        if i % 2:  # 当i为奇数时，提取序列而不是名字
            old_d[j] = histogram(new_list[i][j], old_d[j])
for j in range(41):
    sum1[j] = sum(old_d[j].values())
print(sum(old_d[j].values()))

HX = [0 for i in range(41)]

for j in range(41):
    for key in old_d[j].keys():
        if j != 20:
            HX[j] += -math.log(2, old_d[j].get(key, 0) /60959) * old_d[j].get(key, 0) / 60959 #60959为负样本的数目
print(HX)
import matplotlib.pyplot as plt
fin = open("ptrain.txt")   #负样本训练集
new_list = []  # 新列表
sum1 = [0 for i in range(41)]
for line in fin:
    rs = line.rstrip('\n')  # delete the '\n' after every letter
    new_list.append(rs)  # new_list is used to restore letters


old_d = dict()  # initialize the dictionary
for i in range(0, 41): #0遍历到第40个位置
    old_d[i] = dict();


for i in range(len(new_list)):  # calculate the leter
    for j in range(41):
        if i % 2:  # 当i为奇数时，提取序列而不是名字
            old_d[j] = histogram(new_list[i][j], old_d[j])
for j in range(41):
    sum1[j] = sum(old_d[j].values())
print(sum(old_d[j].values()))

HY = [0 for i in range(41)]
for j in range(41):
    for key in old_d[j].keys():
        if j != 20:
            HY[j] += -math.log(2, old_d[j].get(key, 0) /2559) * old_d[j].get(key, 0) / 2559 #2559为负样本的数目
print(HY)
import matplotlib.pyplot as plt
##画图
plt.rcParams['font.family']=['SimHei']#设置中文字体
X=[]
X=[i-20 for i in range(41)]
X.remove(X[20])
HX.remove(HX[20])
HY.remove(HY[20])
print(HX)
print(HY)
plt.plot(X,HX,'b',label='线1',linewidth=2)
plt.plot(X,HY,'r',label='线1',linewidth=2)
plt.show()