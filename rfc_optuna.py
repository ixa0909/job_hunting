#import
import csv
import numpy as np
from numpy.lib.function_base import average
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import optuna

#評価用
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score,f1_score,recall_score


#optunaで探索する範囲
def objective(trial):
    min_samples_split = trial.suggest_int("min_samples_split", 8, 16)
    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 4, 64, 4))
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    max_depth=int(trial.suggest_discrete_uniform("max_depth", 100, 300, 100))
    n_estimators= int(trial.suggest_discrete_uniform("n_estimators", 50, 200, 50))
    RFC = RandomForestClassifier(min_samples_split = min_samples_split, 
                                max_leaf_nodes = max_leaf_nodes,
                                criterion = criterion,max_depth=max_depth,n_estimators=n_estimators)
    RFC.fit(x_train, y_train)
    return 1.0 - accuracy_score(y_test, RFC.predict(x_test))

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


#機械学習器のハイパーパラメータ調整用
#特徴量データ読み込み　データシャッフル
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
with open("data//y_test_index.csv","r",encoding="utf-8") as fy_test:
    lines=csv.reader(fy_test)
    y_test_read=[row for row in lines]

y_test=[ int(data[0]) for data in y_test_read]
y_test=np.array(y_test)
np.random.seed(10)
np.random.shuffle(y_test)
y_test=y_test.tolist()

# ハイパーパラメータの自動最適化
study = optuna.create_study()
study.optimize(objective, n_trials = 50)

"""
結果
[I 2022-01-14 02:28:44,673] Trial 63 finished with value: 0.5083053726026263 and parameters:
 {'min_samples_split': 12, 'max_leaf_nodes': 64.0, 'criterion': 'gini', 'max_depth': 100.0, 'n_estimators': 200.0}
. Best is trial 63 with value: 0.5083053726026263.
"""

