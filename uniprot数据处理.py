# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 10:44:09 2020

@author: 20170426-2
"""

import requests
import pandas as pd
import os
import json
import random
import time
print(os.getcwd())  # 显示当前路径

# 整理下载的fasta文件 多行并一行
seq = {}         #建立seq字典
file = open('fasta_1.fasta', 'r')
fw = open('fasta_1(1).fasta', 'w')
# 提取第一行前面数据，取消后部分序列的空格
for line in file:
    if line.startswith('>'):
        name = line.split()[0]  # 以空格对字符分割,str.split("#", 1)以#为分隔符，返回2个参数列表
        seq[name] = ''
    else:
        seq[name] += line.replace('\n', '')
file.close()
for i in seq.keys():
    fw.write(i)
    fw.write('\n')
    fw.write(seq[i])
    fw.write('\n')
fw.close()

# 将ID提出来
x = []
fr = pd.read_csv('fasta_1(1).fasta', sep='\n', header=None)
fr_list = list(fr.values)
n = len(fr_list)
for i in range(n):
    if (i + 1) % 2:
        x.append(fr_list[i])
Id = []
m = len(x)
for i in range(m):
    Id.append(x[i][0][4:10])

# 对每一个名字找位点
url = 'https://www.uniprot.org/uniprot/XXX.txt'
Id_site = {}
for t in range(len(Id)):
    target = url.replace('XXX', Id[t])
    p = requests.get(target)
    playFile = open('1.txt', 'wb')
    for chunk in p.iter_content(50000):
        playFile.write(chunk)
    playFile.close()

    df = pd.read_csv('1.txt', sep='\n', header=None, )
    data = df[0]
    cc = []
    for i in range(len(data)):
        if data[i] != '//':
            aa = data[i].split()
            if aa[0] == 'FT' and aa[1] == 'MOD_RES':
                bb = data[i + 1].split()
                if 'Omega-N-methylarginine' in bb[1]:
                    cc.append(aa[2])
    Id_site[Id[t]] = cc
    t += 1

    print(t)

#     print(Id_site.items())
#
# for i in Id_site.values():
#     for j in i:
#         if j in ['P', 'Q', 'A', 'O', 'B']:
#             pro = j
#             for k in range(len(pro)):
#                 if pro[k] == ':':cc
#                     males = filter(lambda y: pro == y[1], i)
#                     j = pro[int(k) + 1:]

for i in Id_site.keys():
    for j in range(len(Id_site[i])):
        if Id_site[i][j][0] in ['P', 'Q', 'A', 'O', 'B']:
            pro = Id_site[i][j]
#             for k in range(len(pro)):
#                 if pro[k] == ':':
#                     print(k)
#                     Id_site[i][j] = pro[int(k) + 1:]
# # for i in Id_site.values():
# #     print(i)
# #
# #                 # 将ID和位置输入
# # # 读取Id_site
# # file = open('Id_site.txt', 'r')文件
# # # js = json.dumps(Id_site)
# # file = open('Id_site.txt', 'w')
# # file.write(js)
# # file.close()
# #
# js = file.read()
# dic = json.loads(js)
# Id_site = list(dic)
# for i in Id_site:
#     aa = []
#     for j in range(len(dic[i])):
#         aa.append(int(dic[i][j]))
# # file.close()
#
# ID_site = {}
#     ID_site[i] = aa

# # 找到去同源后的位点，将字典里的字符转化为数组
# del (Id_site['P53996'])
# Id.remove('P53996')
PS_site = {}
for line in Id:
    PS_site[line] = Id_site[line]
    PS_site[line] = [int(x) for x in PS_site[line]]
# PS_site = Id_site
# 找所有的R位点
K_site = {}
for i in range(int(n / 2)):
    uniprot_id = Id[i]
    l = fr_list[2 * i + 1]
    dd = []
    for j in range(len(l[0])):
        if l[0][j] == 'R':
            dd.append(j + 1)
    K_site[uniprot_id] = dd

# 负样本位点
NS_site = {}
for line in Id:
    c = [x for x in K_site[line] if x not in PS_site[line]]
    NS_site[line] = c

# 切片段，长度不足41的蛋白质序列补0
lenth = 20
squence = list(seq.values())
ee = []
dd = []

for i in range(len(Id)):
    name = list(seq.keys())[i][0:11]
    Max = len(squence[i])
    for site in NS_site[Id[i]]:
        Nsite = site - 1
        if Nsite - lenth < 0:
            nsquence_list = 'O' * lenth + squence[i]
            ee.append(name + 'R|' + str(site))
            NNsite = Nsite + lenth
            ee.append(nsquence_list[NNsite - lenth:NNsite + lenth + 1])
            continue

        if Nsite + lenth > Max - 1:
            nsquence_list = squence[i] + 'O' * lenth
            ee.append(name + 'R|' + str(site))
            ee.append(nsquence_list[Nsite - lenth:Nsite + lenth + 1])
            continue

        if Nsite + lenth > Max - 1 and Nsite - lenth < 0:
            nsquence_list = 'O' * lenth + squence[i] + 'O' * lenth
            ee.append(name + 'R|' + str(site))
            NNsite = Nsite + lenth
            ee.append(nsquence_list[NNsite - lenth:NNsite + lenth + 1])

        else:
            nsquence_list = squence[i]
            ee.append(name + 'R|' + str(site))
            ee.append(nsquence_list[Nsite - lenth:Nsite + lenth + 1])

for i in range(len(Id)):
    name = list(seq.keys())[i][0:11]
    Max = len(squence[i])
    for site in PS_site[Id[i]]:
        Psite = site - 1
        if Psite - lenth < 0:
            squence_list = 'O' * lenth + squence[i]
            dd.append(name + 'R|' + str(site))
            NPsite = Psite + lenth
            dd.append(squence_list[NPsite - lenth:NPsite + lenth + 1])
            continue

        if Psite + lenth > Max - 1:
            squence_list = squence[i] + 'O' * lenth
            dd.append(name + 'R|' + str(site))
            dd.append(squence_list[Psite - lenth:Psite + lenth + 1])
            continue

        if Psite + lenth > Max - 1 and Psite - lenth < 0:
            squence_list = 'O' * lenth + squence[i] + 'O' * lenth
            dd.append(name + 'R|' + str(site))
            NPsite = Psite + lenth
            dd.append(squence_list[NPsite - lenth:NPsite + lenth + 1])

        else:
            squence_list = squence[i]
            dd.append(name + 'R|' + str(site))
            dd.append(squence_list[Psite - lenth:Psite + lenth + 1])

Plist = []
for i in range(int(len(dd) / 2)):
    if len(dd[i * 2 + 1]) == 41 and dd[i * 2 + 1][20] == 'R':
        Plist.append(dd[i * 2])
        Plist.append(dd[i * 2 + 1])

Nlist = []
for i in range(int(len(ee) / 2)):
    if len(ee[i * 2 + 1]) == 41 and ee[i * 2 + 1][20] == 'R':
        Nlist.append(ee[i * 2])
        Nlist.append(ee[i * 2 + 1])

# 打乱序列
aa = list(range(int(len(Plist) / 2)))
random.shuffle(aa)
NPlist = []
for x in aa:
    NPlist.append(Plist[2 * x])
    NPlist.append(Plist[2 * x + 1])

bb = list(range(int(len(Nlist) / 2)))
random.shuffle(bb)
NNlist = []
for x in bb:
    NNlist.append(Nlist[2 * x])
    NNlist.append(Nlist[2 * x + 1])

# aa = random.sample(range(1000, int(len(Plist) / 2)), 40000)
# bb = random.sample(range(1000, int(len(Nlist) / 2)), 20000)
# cho_p = []
# for x in aa:
#     cho_p.append(Plist[2 * x])
#     cho_p.append(Plist[2 * x + 1])
#
# cho_n = []
# for x in bb:
#     cho_n.append(Nlist[2 * x])
#     cho_n.append(Nlist[2 * x + 1])
#
# tp = Plist[0:200]
# tn = Nlist[0:200]
# tp.extend(tn)
#
# dd = NPlist[30000:60000]
#
# P = open('Positive_S(30)1.txt', 'a+')
# N = open('ntrain_S(30).txt', 'a+')
#
# for line in cho_p:
#     P.write(line + '\n')
#
# for line in cho_n:
#     N.write(line + '\n')
#
# P.close()
# N.close()
#
# T = open('test.txt', 'a+')
# for line in test:
#     T.write(line + '\n')
# T.close()
#
# # 导入数据
# file_name1 = open('D:\guoxinyun\data\phosphorylation\Positive_S(30)(30).txt', 'r')
# file_name2 = open('D:\guoxinyun\data\phosphorylation/Nagetive_S(30)(30).txt', 'r')
#
# Plist = []
# Nlist = []
#
# ee = []
# yy = ee[0:100]
# mm = []
#
# for lines in file_name1:
#     line = lines.strip('\n')
#     Plist.append(line)
# file_name1.close()
#
# for lines in file_name2:
#     line = lines.strip('\n')
#     Nlist.append(line)
# file_name2.close()


# 分测试集20%，正负训练集80%
ptest = NPlist[:1280]#列表从0开始索引，左闭右开
ntest = NNlist[:34528]
test = ptest + ntest
T = open('test.txt', 'w+')
for line in test:
    T.write(line + '\n')
T.close()

ptrain = NPlist[1280:]
ntrain = NNlist[34528:]

P = open('ptrain.txt', 'w+')
N = open('ntrain.txt', 'w+')

for line in ptrain:
    P.write(line + '\n')

for line in ntrain:
    N.write(line + '\n')

P.close()
N.close()
