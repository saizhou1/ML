# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :code
# @File     :SMOTE
# @Date     :2021/5/7 17:25
# @Author   :sai
# @Software :PyCharm
-------------------------------------------------
"""
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
from collections import Counter

train_data = pd.read_csv('AAC.txt', sep=',', header=None)
train_data = np.vstack((train_data.iloc[:2559, 1:].values, train_data.iloc[2559:7559, 1:].values))
train_label = np.array([1 for i in range(2559)] + [0 for i in range(5000)])

smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(train_data, train_label )
print(Counter(y_smo))