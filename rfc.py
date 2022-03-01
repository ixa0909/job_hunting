#import
import csv
import numpy as np
from numpy.lib.function_base import average
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#訓練用
#特徴量データ読み込み　データシャッフル
with open("data//x_train.csv","r",encoding="utf-8") as fx_train:
    lines=csv.reader(fx_train,delimiter=",")
    x_train=[]
    for line in lines:
        x_train.append(line)
        
x_train=np.array(x_train)
np.random.seed(10)
np.random.shuffle(x_train)
x_train=x_train.astype(np.float64).tolist()

#教師データ読み込み　データシャッフル
with open("data//y_train_index.csv","r",encoding="utf-8") as fy_train:
    lines=csv.reader(fy_train)
    y_train_read=[row for row in lines]

y_train=[ int(data[0]) for data in y_train_read]
y_train=np.array(y_train)
np.random.seed(10)
np.random.shuffle(y_train)
y_train=y_train.tolist()

"""
#test data
with open("../data/x_test.csv","r",encoding="utf-8") as fx_test:
    lines=csv.reader(fx_test,delimiter=",")
    x_test=[]
    for line in lines:
        x_test.append(line)
        
x_test=np.array(x_test)
np.random.seed(10)
np.random.shuffle(x_test)
x_test=x_test.astype(np.float64).tolist()

with open("../data/y_test_index.csv","r",encoding="utf-8") as fy_test:
    lines=csv.reader(fy_test)
    
    y_test_read=[row for row in lines]
y_test=[ int(data[0]) for data in y_test_read]

y_test=np.array(y_test)
np.random.seed(10)
np.random.shuffle(y_test)
y_test=y_test.tolist()
"""
clf=RandomForestClassifier(n_estimators=200,max_depth=100,criterion="gini",min_samples_split=12,max_leaf_nodes=64)

#学習
clf.fit(x_train,y_train)

#学習済みモデルの保存
import pickle
with open("rfc.pkl",mode="wb") as f:
    pickle.dump(clf,f,protocol=2)

#pred=clf.predict(x_test)
