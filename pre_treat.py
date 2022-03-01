# -*- coding: utf-8 -*-
import pandas as pd

#実行は半年分で約15~20分間かかる
#訓練用は１回眠れる

#機械学習器の訓練データの前処理
"""
race_data = pd.read_pickle("data//race_data_2016a.pkl")
race_data_2016_b = pd.read_pickle("data//race_data_2016b.pkl")
race_data_2017_a = pd.read_pickle("data//race_data_2017a.pkl")
race_data_2017_b = pd.read_pickle("data//race_data_2017b.pkl")
race_data_2018_a = pd.read_pickle("data//race_data_2018a.pkl")
race_data_2018_b = pd.read_pickle("data//race_data_2018b.pkl")
race_data_2019_a = pd.read_pickle("data//race_data_2019a.pkl")
race_data_2019_b = pd.read_pickle("data//race_data_2019b.pkl")
race_data_2020_a = pd.read_pickle("data//race_data_2020a.pkl")
race_data_2020_b = pd.read_pickle("data//race_data_2020b.pkl")


race_data=race_data.append(race_data_2016_b)
race_data=race_data.append(race_data_2017_a)
race_data=race_data.append(race_data_2017_b)
race_data=race_data.append(race_data_2018_a)
race_data=race_data.append(race_data_2018_b)
race_data=race_data.append(race_data_2019_a)
race_data=race_data.append(race_data_2019_b)
race_data=race_data.append(race_data_2020_a)
race_data=race_data.append(race_data_2020_b)
"""

#機械学習器のハイパーパラメータ調整用データの前処理
race_data = pd.read_pickle("data//race_data_2021a.pkl")
race_data_2021_b = pd.read_pickle("data//race_data_2021b.pkl")

race_data=race_data.append(race_data_2021_b)


#-----------------
#各レースに関するデータが順位結果順になっているので車番順に変更
for line in range(0,len(race_data),9):
  race_data[line:line+9]=race_data[line:line+9].sort_values("車番")


#出走表における「選手名府県/年齢/期別」のデータを分ける
race_data = pd.concat([race_data, race_data["選手名府県/年齢/期別"].str.split("/", expand=True)], axis=1).drop("選手名府県/年齢/期別", axis=1)
race_data = race_data.rename(columns={0:"選手名府県",1: "年齢", 2: "期別"})
race_data = pd.concat([race_data, race_data["選手名府県"].str.split(" ",n=2, expand=True)], axis=1).drop(["選手名府県",0,1], axis=1)
race_data = race_data.rename(columns={ 2: "県"})



#--------------------
#同じ県の人が何人いるかのデータを作成
for i in range(0,len(race_data["県"]),9):
  data=race_data["県"][i:i+9]
  a=data.tolist()
  #print(race_data["県"][i:i+9])
  for j in range(i,i+9):
    if a[j%9] is None:
      race_data["県"][j]=0
    else:
      race_data["県"].iat[j]=a.count(a[j%9])-1


#文字データを数値化
race_data.replace({'級班':{"nan":0.0,"A1":1/64,"A2":1/32, "A3":1/16, "L1":1/8, "S1":1/4, "S2":1/2, "SS":1}},inplace=True)
race_data.replace({'好気合':{"nan":0.0,"★":1.0}},inplace=True)
race_data.fillna(0,inplace=True)
race_data.replace({'予想':{"nan":0.0, "×":0.0, "△":1/8, "▲":1/4, "○":1/2, "◎":1.0, "注":0}},inplace=True)

#----------------
#追加
race_data = race_data.rename(columns={"逃":"逃戦"})

#以下は主にダミー変数化、ダミー化変数の対象と，カテゴリーを定義
dummy_targets = {"脚質": ["両", "追", "逃"]}

#定義したカテゴリーを指定しておく
for key, item in dummy_targets.items():
    race_data[key] = pd.Categorical(race_data[key], categories=item)

#ダミー変数化されたデータフレームを格納するリストと削除する列のリスト宣言
dummies = [race_data]
drop_targets = []

#ダミー変数化してdummiesに代入
for key, items in dummy_targets.items():
    dummy = pd.get_dummies(race_data[key])
    dummies.append(dummy)
    drop_targets.append(key)

#ダミー変数化されたデータフレームを大元のデータフレームに結合
race_data = pd.concat(dummies, axis=1).drop(drop_targets,  axis=1)

#落車などで順位が出なかった部分を9位として変換
race_data = race_data.replace({"着順":{"失":"9", "落":"9", "故":"9", "欠":"9"}})
#ギヤ倍数の例外表示を変換
race_data["ギヤ倍数"] = race_data["ギヤ倍数"].map(lambda x: x[:4] if type(x)!=int and len(x)>4 else x)
#期別に含まれる欠車の文字を除外
race_data["期別"] = race_data["期別"].map(lambda x: x.replace(" （欠車）", "") if type(x)!=float and type(x)!=int and "欠車"in x else x)

#-----------------
#着順の列を3着以内は1,それ以外は0に変換
race_data["着順"] = race_data["着順"].map(lambda x: 1 if x in ["1"]  else 0.0)

#全データをfloat型に変換
race_data = race_data.astype("float64",errors="raise")

#特徴量データ（出走表）と教師データ（レース結果）に分割
race_y = race_data['着順']
race_x = race_data.drop('着順', axis=1)
race_x = race_x.loc[:, ["車番",	"予想",	"好気合",	"総評",	"枠番",	"級班",	"ギヤ倍数"	,"競走得点"	,"S","B","逃戦","捲"	,"差",	"マ","1着"	,"2着",	"3着"	,"着外",	"勝率",	"2連対率",	"3連対率",	"年齢",	"期別","県",	"両",	"追",	"逃"]]

#-------------------
#まだnanが残っている？バグ処理
race_x.fillna(0,inplace=True)

"""
#データを保存
race_x.to_csv("data//pre_x_train.csv")
race_y.to_csv("data//pre_y_train.csv")
"""

#機械学習器のハイパーパラメータ調整用データ
race_x.to_csv("data//pre_x_test.csv")
race_y.to_csv("data//pre_y_test.csv")
