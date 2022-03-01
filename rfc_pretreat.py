#import
import torch
import torch.nn as nn
import numpy as np
import math

#注意：：動かした後にファイル内の[]をテキストエディタの置換機能で消す！！

#特徴量データと教師データ
l_x=[]
l_y=[]

#作成した機械学習器の入力は多次元配列に対応していないので、レースデータを多次元配列　→　１次元配列に変更
with open("data//pre_y_test.csv",mode="r",encoding="utf-8") as file:
    l=len(file.readlines())

#特徴量データ読み込み
with open("data//pre_x_test.csv",mode="r",encoding="utf-8") as file:
    count=0
    for lines in file.readlines():
        count+=1
        if count==1:
            continue
        elif count==l+1:
            break

        line=lines.replace("\n","").split(",")[1:]

        #例外発生対応
        if '' in line:
            print(count)
            print(line)
            print(len(line))
        l_x.append(list(map(float,line)))

#教師データ読み込み
with open("data//pre_y_test.csv",mode="r",encoding="utf-8") as file:
    count=0
    for lines in file.readlines():
        count+=1
        if count==1:
            continue
        elif count==l+1:
            break
        line=lines.replace("\n","").split(",")[1:]
        l_y.append(list(map(float,line)))

count=0

#特徴量データを多次元配列　→　１次元配列に変更
for i in range(0,len(l_x),9):
    for j in range(1,9):
        l_x[i].extend(l_x[i+j])
x_train=[]
for i in range(0,len(l_x)):
    if len(l_x[i])>30:
        x_train.append(l_x[i])

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
c0=[1,0,0,0,0,0,0,0,0]
c1=[0,1,0,0,0,0,0,0,0]
c2=[0,0,1,0,0,0,0,0,0]
c3=[0,0,0,1,0,0,0,0,0]
c4=[0,0,0,0,1,0,0,0,0]
c5=[0,0,0,0,0,1,0,0,0]
c6=[0,0,0,0,0,0,1,0,0]
c7=[0,0,0,0,0,0,0,1,0]
c8=[0,0,0,0,0,0,0,0,1]

count=0
del_index=[]

for data_y in y_train:
    if data_y==c0:
        y_train[count]=0
    elif data_y==c1:
        y_train[count]=1
    elif data_y==c2:
        y_train[count]=2
    elif data_y==c3:
        y_train[count]=3
    elif data_y==c4:
        y_train[count]=4
    elif data_y==c5:
        y_train[count]=5
    elif data_y==c6:
        y_train[count]=6
    elif data_y==c7:
        y_train[count]=7
    elif data_y==c8:
        y_train[count]=8
    else:
        del_index.append(count) 

    count=count+1

#データを削除するとインデックスにずれが生じる対策
number_del=0
for i in del_index:
    del y_train[i-number_del]
    del x_train[i-number_del]
    number_del=number_del+1

#テストデータをファイルに書き込み
with open("data//x_test.csv","w") as f:
    for data in x_train:
        f.write("{0}\n".format(data))

with open("data//y_test_index.csv","w") as f:
    for data in y_train:
        f.write("{0}\n".format(data))

"""
#訓練データをファイルに書き込み
with open("data//x_train.csv","w") as f:
    for data in x_train:
        f.write("{0}\n".format(data))

with open("data//y_train_index.csv","w") as f:
    for data in y_train:
        f.write("{0}\n".format(data))
"""