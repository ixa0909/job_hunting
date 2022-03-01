#import
import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.model_selection import cross_val_score

#注意：：動かした後にファイル内の[]をテキストエディタの置換機能で消す！！

#特徴量データと教師データ
l_y=[]

#作成した機械学習器の入力は多次元配列に対応していないので、レースデータを多次元配列　→　１次元配列に変更
with open("data//pre_y_test.csv",mode="r",encoding="utf-8") as file:
    count=0
    l=len(file.readlines())

#教師データ読み込み
with open("data//pre_y_test.csv",mode="r",encoding="utf-8") as file:
    count=0
    for lines in file.readlines():
        count=count+1
        if count==1:
            continue
        elif count==l+1:
            break
        line=lines.replace("\n","").split(",")[1:]
        l_y.append(list(map(float,line)))


#教師データを多次元配列　→　１次元配列に変更
for i in range(0,len(l_y),9):
    for j in range(1,9):
        l_y[i].extend(l_y[i+j])
y_train=[]
for i in range(0,len(l_y)):
    if len(l_y[i])>2:
        y_train.append(l_y[i])

#１位が複数人いるデータを今回は削除
#教師データとしてあり得るもの
c0=[1,0,0,0,0,0,0,0,0];c1=[0,1,0,0,0,0,0,0,0];c2=[0,0,1,0,0,0,0,0,0];c3=[0,0,0,1,0,0,0,0,0];c4=[0,0,0,0,1,0,0,0,0];c5=[0,0,0,0,0,1,0,0,0];c6=[0,0,0,0,0,0,1,0,0];c7=[0,0,0,0,0,0,0,1,0];c8=[0,0,0,0,0,0,0,0,1]

count=0
del_index=[]
for data_y in y_train:
    if data_y!=c0 and data_y!=c1 and data_y!=c2 and data_y!=c3 and data_y!=c4 and data_y!=c5 and data_y!=c6 and data_y!=c7 and data_y!=c8:
        del_index.append(count)
    count=count+1

#データを削除するとインデックスにずれが生じる対策
number_del=0
for i in del_index:
    del y_train[i-number_del]
    number_del=number_del+1

#fileに書き込み
with open("data//y_test_nn.csv","w") as f:
    for data in y_train:
        f.write("{0}\n".format(data))