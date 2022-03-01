# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup as bs4
from construct import len_
import requests
import re
import time

import torch
import torch.nn as nn
import numpy as np
import math

from numpy.lib.function_base import average
from sklearn.ensemble import RandomForestClassifier

import traceback
import pandas as pd
import pickle

main_colum = ['予想', '好気合', '総評', '枠番', '車番', '選手名府県/年齢/期別', '級班', '脚質', 'ギヤ倍数', '競走得点', 'S','B','逃','捲','差','マ','1着', '2着', '3着', '着外','勝率','2連対率','3連対率']

def scrape(url):
        try:
            #htmlを取得
            req=requests.get(url)
            soup = bs4(req.content, 'html.parser')

            #html内の表データをDataFrameのリストで取得
            main = pd.read_html(url)

            #出走表の表だけを取得
            df = main[0][:-1]
            df.columns = main_colum

            #str型に変換
            df = df.astype(str)

        #各例外に対応してメッセージを出力
        except IndexError:
            print('IndexError: {}', url)

        except KeyError:
            print('keyerror: {}', url)

        except ValueError:
            print("ValueError: {}", url)

        except :
            traceback.print_exc()

        return df

#入力例
"""
https://keirin.kdreams.jp/kochi/racedetail/7420220201010003/?l-id=l-pc-srdi-srdi-raceinfo_kaisai_detail_race_nav_btn-3
"""

def predict_result(url):
    results = scrape(url)

    #-----------------
    #レース出場者が9人に未たない時に空データを加える
    key=results.keys()

    len_frame=len(results[key].index)

    if len_frame<9:
        vacant_pd=pd.DataFrame(index=range(9-len_frame))
        results=results.append(vacant_pd,ignore_index=True)

    race_data=results

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
        for j in range(i,i+9):
            if type(a[j%9]) is float:
                race_data["県"][j]=0
            else:
                race_data["県"].iat[j]=a.count(a[j%9])-1


    #文字データを数値化
    race_data.replace({'級班':{"nan":0.0,"A1":1/64,"A2":1/32, "A3":1/16, "L1":1/8, "S1":1/4, "S2":1/2, "SS":1}},inplace=True)
    race_data.replace({'好気合':{"nan":0.0,"★":1.0}},inplace=True)
    race_data.fillna(0,inplace=True)
    race_data.replace({'予想':{"nan":0.0, "×":0.0, "△":1/8, "▲":1/4, "○":1/2, "◎":1.0, "注":0}},inplace=True)

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

    #ギヤ倍数の例外表示を変換
    race_data["ギヤ倍数"] = race_data["ギヤ倍数"].map(lambda x: x[:4] if type(x)!=int and len(x)>4 else x)
    #期別に含まれる欠車の文字を除外
    race_data["期別"] = race_data["期別"].map(lambda x: x.replace(" （欠車）", "") if type(x)!=float and type(x)!=int and "欠車"in x else x)

    #全データをfloat型に変換
    race_data = race_data.astype("float64",errors="raise")

    race_x = race_data.loc[:, ["車番",	"予想",	"好気合",	"総評",	"枠番",	"級班",	"ギヤ倍数"	,"競走得点"	,"S","B","逃戦","捲"	,"差",	"マ","1着"	,"2着",	"3着"	,"着外",	"勝率",	"2連対率",	"3連対率",	"年齢",	"期別","県",	"両",	"追",	"逃"]]

    #-------------------
    #まだnanが残っている？バグ処理
    race_x.fillna(0,inplace=True)


    race_x=race_x.to_numpy().tolist()

    #作成した機械学習器の入力は多次元配列に対応していないので、レースデータを多次元配列　→　１次元配列に変更
    for i in range(0,len(race_x),9):
        for j in range(1,9):
            race_x[i].extend(race_x[i+j])
    x_train=[]
    for i in range(0,len(race_x)):
        if len(race_x[i])>30:
            x_train.append(race_x[i])


##nnを定義しなおす

#model_path="model.pth"
#model.load_state_dict(torch.load(model_path))


    #学習済みモデルをロード
    with open("rfc.pkl",mode="rb") as f:
        clf=pickle.load(f)

    return clf.predict(x_train)[0]

if __name__=="__main__":
    url=input("url\n")
    print(predict_result(url))
