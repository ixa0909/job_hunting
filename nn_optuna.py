#import
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import optuna

#評価
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,f1_score,recall_score

#ReLU関数を定義
relu=nn.ReLU()

#sigmoid関数を定義
def sigmoid(x):
  return 1.0 / (1.0 + math.e**-x)

#softmax関数を定義
softmax=nn.Softmax(dim=1)


#class MLP
class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.l2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    h = relu(self.l1(x))
    return softmax(self.l2(h))

#optunaで探索する範囲
def objective(trial):
    hide_node = trial.suggest_int("hide_node", 96, 512,8)
    lr = trial.suggest_discrete_uniform("lr", 1e-4, 1e-3,q=1e-4)
    model = MLP(243, hide_node, 9)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = torch.nn.MSELoss()

    max_epoch = 1000
    for epoch in range(max_epoch):
        pred = model(x_train)
        error = criterion(y_train, pred)
        model.zero_grad()
        error.backward()
        optimizer.step()

    pred=model(x_test)
    pred_index=[]

    for i in range(len(pred)):
      pred_index.append(torch.argmax(pred[i]))

    return 1.0 - accuracy_score(y_test_index,pred_index)

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
with open("data//y_train_nn.csv","r",encoding="utf-8") as fy_train:
    lines=csv.reader(fy_train,delimiter=",")
    y_train=[]
    for line in lines:
        y_train.append(line)

y_train=np.array(y_train)
np.random.seed(10)
np.random.shuffle(y_train)
y_train=y_train.astype(np.float64).tolist()


#機械学習器のハイパーパラメータ調整用
#特徴データ読み込み　データシャッフル
with open("data//x_test.csv","r",encoding="utf-8") as fx_test:
    lines=csv.reader(fx_test,delimiter=",")
    x_test=[]
    for line in lines:
        x_test.append(line)

x_test=np.array(x_test)
np.random.seed(10)
np.random.shuffle(x_test)
x_test=x_test.astype(np.float64).tolist()

#教師データ読み込み　データシャッフル
with open("data//y_test_index.csv","r",encoding="utf-8") as fy_test_index:
    lines=csv.reader(fy_test_index)

    y_test_index_read=[row for row in lines]
y_test_index=[ int(data[0]) for data in y_test_index_read]

y_test_index=np.array(y_test_index)
np.random.seed(10)
np.random.shuffle(y_test_index)
y_test_index=y_test_index.tolist()

#正答率確認用にランダムフォレスト用の教師データを読み込み
with open("data//y_train_index.csv","r",encoding="utf-8") as fy_train_index:
    lines=csv.reader(fy_train_index)

    y_train_index_read=[row for row in lines]

y_train_index=[ int(data[0]) for data in y_train_index_read]

y_train_index=np.array(y_train_index)
np.random.seed(10)
np.random.shuffle(y_train_index)
y_train_index=y_train_index.tolist()


x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)

x_test = torch.tensor(x_test, dtype=torch.float)

# ハイパーパラメータの自動最適化
study = optuna.create_study()
study.optimize(objective, n_trials = 50)

"""
例）
[I 2022-03-02 04:13:50,003] Trial 2 finished with value: 0.544757880783729 and parameters: 
{'hide_node': 328, 'lr': 0.0002}. Best is trial 2 with value: 0.544757880783729.
"""
