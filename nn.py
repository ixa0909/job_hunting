#import
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import accuracy_score
import math

#ReLU関数を定義
relu=nn.ReLU()

#sigmoid関数を定義
def sigmoid(x):
  return 1.0 / (1.0 + math.e**-x)

#softmax関数を定義
softmax=nn.Softmax(dim=1)


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

#正答率確認用にランダムフォレスト用の教師データを読み込み
with open("data//y_train_index.csv","r",encoding="utf-8") as fy_train_index:
    lines=csv.reader(fy_train_index)

    y_train_index_read=[row for row in lines]
y_train_index=[ int(data[0]) for data in y_train_index_read]

y_train_index=np.array(y_train_index)
np.random.seed(10)
np.random.shuffle(y_train_index)
y_train_index=y_train_index.tolist()

#データをtensor型に変更
x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float)


#class MLP
class MLP(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(MLP, self).__init__()
    self.l1 = nn.Linear(input_size, hidden_size)
    self.l2 = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    h = relu(self.l1(x))
    return softmax(self.l2(h))

model = MLP(243, 128, 9)

optimizer = torch.optim.Adam(model.parameters(),lr=0.0003)
criterion = torch.nn.MSELoss()

history_loss = []
history_accuracy=[]

max_epoch = 1000

def copy(a):
	return a

for epoch in range(max_epoch):
    pred = model(x_train)
    pred_index=[]
    
    for i in range(len(pred)):
        pred_index.append(torch.argmax(pred[i]))

    #history_accuracy.append(accuracy_score(y_train_index,pred_index))
    error = criterion(y_train, pred)
    model.zero_grad()
    error.backward()
    optimizer.step()
    #history_loss.append(error.item())

#history_loss = np.array(history_loss, dtype=np.float32)
#epochs = np.arange(1, max_epoch+1)

#学習済みモデルの保存
model_path="nn.pth"
torch.save(model.state_dict(),model_path)
#pred=model(x_test)
#print(torch.argmax(pred))
