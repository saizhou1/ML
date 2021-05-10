# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 09:07:48 2020

@author: 20170426-2
"""
import os
import pandas as pd
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

os.chdir('D:\guoxinyun\RF')

#导入数据
file_name1 = open('D:\guoxinyun\data\phosphorylation\ptrain_S(30)(30).txt','r')

file_name2 = open('D:\guoxinyun\data\phosphorylation/ntrain_S(30)(30).txt','r')



Plist = []
Nlist = []

for lines in file_name1:
    line=lines.strip('\n')
    Plist.append(line)
file_name1.close()

for lines in file_name2:
    line=lines.strip('\n')
    Nlist.append(line)
file_name2.close()

T = open('D:\\guoxinyun\\data\\phosphorylation\\test.txt','r')
test = []
for lines in T:
    line=lines.strip('\n')
    test.append(line)
T.close()

        
aa = list(range(int(len(Plist)/2)))
random.shuffle(aa)
NPlist=[]
for x in aa:
    NPlist.append(Plist[2*x])
    NPlist.append(Plist[2*x+1])


#各随机选10000条
new_Plist=[]
new_Nlist=[]

for i in range(len(Plist)):
    if (i%2):
        new_Plist.append(Plist[i])
        
        
for i in range(len(Nlist)):
    if (i%2):
        new_Nlist.append(Nlist[i])

new_test=[]        
new_test_pos = []
new_test_nag = []
for i in range(len(test)):
    if (i%2) :
        new_test.append(test[i]) 
new_test_pos =  new_test[:1000] 
new_test_nag =   new_test[1000:]  
    

ee=[]
dd=[]
str1='OARNDCQEGHILKMFPSTWYVU'
for item in new_test_pos:
    aa=np.zeros((41,22))
    for i in range(len(item)):
        b = str1.index(item[i])
        aa[i][b]=1    
    a = aa.flatten()
    new_a = np.append(a,1)
    ee.append(new_a)
    
    
    
for item in new_test_nag:
    bb=np.zeros((41,22))
    for i in range(len(item)):
        b = str1.index(item[i])
        bb[i][b]=1
    b = bb.flatten()
    new_b = np.append(b,0)    
    dd.append(new_b)    

ind = ee + dd
    

pos = random.sample(new_Plist,10000)
nag = random.sample(new_Nlist,10000)

pos_n = random.sample(new_Plist,3000)
nag_n = random.sample(new_Nlist,20000)

pos_test = random.sample(new_Plist,200)
nag_test = random.sample(new_Nlist,200)

test = pos_test + nag_test

P = open('NPositive.txt','a+')
N = open('NNagetive.txt','a+')
T = open('test.txt','a+')

for line in pos_n:
    P.write(line+'\n')
    
for line in nag_n:
    N.write(line+'\n')
    
for line in test:
    T.write(line+'\n')

P.close()
N.close()
T.close()

#二进制编码
pos_new=[]
nag_new=[]
str1='OARNDCQEGHILKMFPSTWYVU'
for item in new_Plist:
    aa=np.zeros((41,22))
    for i in range(len(item)):
        b = str1.index(item[i])
        aa[i][b]=1    
    a = aa.flatten()
    new_a = np.append(a,1)
    pos_new.append(new_a)
    
    
    
for item in new_Nlist:
    bb=np.zeros((41,22))
    for i in range(len(item)):
        b = str1.index(item[i])
        bb[i][b]=1
    b = bb.flatten()
    new_b = np.append(b,0)    
    nag_new.append(new_b)    

    


#随机选
nag_new = random.sample(nag_new, len(pos_new))

#k-means选
X = np.array(nag_new) 
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
class_ = kmeans.labels_ 
class_n = class_.tolist()
center = kmeans.cluster_centers_ 

P = np.array(pos_new)
center_P = P.mean(axis=0)
dist = np.linalg.norm(center_P - center[1])

nag_new = random.sample(class1, len(pos_new))

class3=[]
for i in range(len(class_n)):
    if class_n[i]==2:
        class3.append(nag_new[i])
        

#分训练集和测试集
n= len(pos_new)
offnum1 = int(n*0.8)
offnum2 = int(n*0.1)
pos_train = pos_new[:offnum1]
pos_test = pos_new[offnum1:offnum1+offnum2]
pos_ver = pos_new[offnum1+offnum2:]

n= len(nag_new)
offnum1 = int(n*0.8)
offnum2 = int(n*0.1)
nag_train = nag_new[:offnum1]
nag_test = nag_new[offnum1:offnum1+offnum2]
nag_ver = nag_new[offnum1+offnum2:]

nag_test = random.sample(nag_new, len(pos_test))

#随机森林训练

train = pos_train + nag_train
train_arr = np.matrix(train)
test = pos_test + nag_test
test_arr = np.matrix(test)

ver = pos_ver + nag_ver
ver_arr = np.matrix(ver)

ind_arr = np.matrix(ind)

X_train = pos_arr[:, :len(test[0])-1]
Y_train = train_arr[:, len(test[0])-1]

X_test = test_arr[:, :len(test[0])-1]
Y_test = test_arr[:, len(test[0])-1]

X_ver = ver_arr[:, :len(ver[0])-1]
Y_ver = ver_arr[:, len(ver[0])-1]

X_ind = ind_arr[:, :len(ver[0])-1]
Y_ind = ind_arr[:, len(ver[0])-1]

#X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

n_estimators_range = [200]
max_depth_range = [5, 6, 7, 8 ,9 ,10]
max_features_range = [5, 10, 12]
result = np.zeros((18,9))
i=0
 
for n_estimators in n_estimators_range:
    for max_depth in max_depth_range:
        for max_features in max_features_range:
            j=0
            clf = RandomForestClassifier(n_estimators=n_estimators , 
                                         max_depth=max_depth,
                                         max_features=max_features,
                                         n_jobs=3,
                                         random_state=0
                                         )
            clf.fit(X_train, Y_train)
    #scores = cross_val_score(clf,X_train, Y_train)#交叉验证
            score =clf.score(X_train, Y_train)
            prediction = clf.predict(X_test) 
    
    
            TP = 0
            FP = 0
            TN = 0
            FN = 0
    
            n=len(prediction)
    
            for t in range(n):
                if Y_test[t] == 1:
                    if prediction[t] == 0:
                        FN = FN+1
                    else:
                        TP = TP+1
                else:
                    if prediction[t]==1:
                        FP = FP+1
                    else:
                        TN = TN+1
    
            acc = (TP+TN)/n
            Sn = TP/(TP+FN)
            Sp = TN/(TN+FP)
            MCC = (TP*TN-FP*FN)/np.sqrt((TP+FN)*(FN+FP)*(TP+FP)*(TN+FN))
            pre_pro = clf.predict_proba(X_test)
            auct = metrics.roc_auc_score(Y_test, pre_pro[:, 1])
            result[i][j]=n_estimators
            j+=1
            result[i][j]=max_depth
            j+=1
            result[i][j]=max_features
            j+=1
            result[i][j]=score
            j+=1
            result[i][j]=acc
            j+=1
            result[i][j]=Sn
            j+=1
            result[i][j]=Sp
            j+=1
            result[i][j]=MCC
            j+=1
            result[i][j]=auct
            i+=1
        
pre_pro = clf.predict_proba(X_ver)
fpr, tpr, _ = roc_curve(Y_ver, pre_pro[:, 1])   
roc_auc = auc(fpr, tpr)  


#独立测试
clf = RandomForestClassifier(n_estimators=300 , 
                                         max_depth=7,
                                         max_features=12,
                                         n_jobs=3,
                                         random_state=0
                                         )
prediction = clf.predict(X_ind)
TP = 0
FP = 0
TN = 0
FN = 0
    
n=len(prediction)
    
for t in range(n):
    if Y_ind[t] == 1:
        if prediction[t] == 0:
            FN = FN+1
        else:
            TP = TP+1
    else:
        if prediction[t]==1:
            FP = FP+1
        else:
            TN = TN+1


#roc曲线画图，计算auc  

import matplotlib.pyplot as plt
pre_pro = clf.predict_proba(X_test)
fpr, tpr, _ = roc_curve(Y_test, pre_pro[:, 1])   
roc_auc = auc(fpr, tpr) 
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
