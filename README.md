ご覧頂きありがとうございます<br>
Thanks for your visiting here.<br>

### 概要（３００字以内）+ α
　競輪の1位予測システムと簡易的Webアプリを開発しました。具体的には、過去のレースデータを取得し、機械学習手法を用いることで作成しました。まず、作成期間を4カ月と定め、優先順や並行的にできる作業を見極めた上で、効率的に速くできるように計画を考え実行しました。取り組む中で工夫として個々の作業に制限時間を設け、段階的な作成や改善により効率を上げ、第３者視点でもコードの見やすさや開発物を考えることで質をより高められるよう努力しました。また、アピールできるのは自作アルゴリズムや多くの新しい知識を短期間で学び応用したことです。まだまだ改善の余地はあります。
<br>


|プログラミング言語 | Python3、html、CSS|
| :---------- | :---- |
|**作成期間**                | **６カ月　(2021年)**|
|**技術**          |**機械学習、Webスクレイピング、アプリ開発**|
|**ツール**          |**Atom、VScode、Excel、Flask**|


&emsp;&emsp;![system](img/system.png)

<font color="Red">
※
</font>入力は出走表掲載のWebページのURL　（ただし、「楽天Kドリームス」に限る）

例）https://keirin.kdreams.jp/kochi/racedetail/7420220201010003/?l-id=l-pc-srdi-srdi-raceinfo_kaisai_detail_race_nav_btn-3

<br>

## 目次　　　　　　 &emsp; &emsp;　 作成の流れ
* 概要　     &emsp;&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;&nbsp;&nbsp;1. 計画
* 作成の流れ　&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &emsp;&thinsp;2. データセットの構築
* ファイル説明　&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &thinsp;3. 機械学習モデルの作成
* Webアプリ化　&emsp; &emsp; &emsp; &emsp; &emsp; &emsp; &thinsp;4. 学習器の検証や精度向上
* 苦労した点(2つ)　&emsp; &emsp; &emsp; &emsp; &emsp; &thinsp;5. 簡易的なWebアプリ化
* アピールできる点(2つ)
* 成長したこと(２つ)
* 参考・引用元



 
&emsp;&emsp;![plan](img/plan.png)

### ファイル説明
|  ファイル、ディレクトリ（/）  | 用途 |
| :---------- | :---- |
|web_scrape.py | データの取得 |
|       data/        | データ |
|rfc.py　         | 機械学習 ランダムフォレスト |
|nn.py　          | 機械学習　ニューラルネットワーク MLP3層 |
|rfc_optuna.py,nn_optuna.py| 機械学習パラメータ調整 |
|rfc.pkl,nn.pth| 学習済みモデル |
|predict_result.py| アプリでの結果予測（python） |
|app/| 簡易的アプリ(html,css) |

<br>
<font color="Red">
※
</font>
主なファイルのみ説明し、細かい処理用や比較的重要でないファイルについての説明は省略しています。
<br>
<br>
<font color="Red">
※
</font>
データの取得のためのWebスクレイピングのコードは[1]を主に参考にしています。Webスクレイピングのコード内においては、独自のコードであることを主張したい箇所にはコメントアウトを２行記述し、<br>

```Python　

#-----------------
#レース出場者が9人に未たない時に空データを加える
for key in results.keys():
      len_frame=len(results[key].index)
      if len_frame<9:
          vacant_pd=pd.DataFrame(index=range(9-len_frame))
          results[key]=results[key].append(vacant_pd,ignore_index=True)

#その日の天気データを取得
#試し
s=str(soup.find_all("p",class_="weather"))
print(s[s.find("雨"):])

```
<br>のようにコードの上に2行のコメントアウトをしています。コメントアウトが１行のところやないところはそれに該当しません。
<br>

### Webアプリ化
　簡易的ではありますがWebアプリを開発しました。画像を載せておきます。

<br>

&emsp;&emsp;![app1](img/app1.png)

&emsp;&emsp;![2](img/app2.png)

&emsp;&emsp;![3](img/app3.png)

<br>

### 主に苦労した2点
* **読みやすいコードを書くこと**<br>
　他人や再度自分が観ることを想定して、アルゴリズムやコードを短いものにしたり、複雑になりすぎないように考えながら取り組んだ。

* **コーディングする上での一般的な苦労**<br>
　短期間での新しい技術の習得とその応用や例外処理、エラーへの対応に最も苦労した。

<br>

### アピールできる2点
* **自作アルゴリズム**<br>
　競輪ではレースに同じ県所属の選手が何人いるかが重要な要素の１つなので、それをデータとして得るためのアルゴリズムです。
```Python
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
```
<br>　同着１位がいるレースのデータを除外するアルゴリズムです。<br>

```Python
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
```

<br>

* **効率的で速いシステム作成**<br>
作成期間を4カ月と定めていたので、優先順や並行的にできる作業を見極めた上で、効率的に速くできるように計画を考え実行しました。また、工夫として個々の作業に制限時間を設け、段階的な作成や改善により効率を上げられるようにしました。




<br>

### 成長したこと(2つ)
* 計画力・作業を効率的に速くこなす力
* システムの質を高めるための視点変換

<br>

### 参考・引用元について
[1]Webスクレイピングのためのコードの参考<br>
https://qiita.com/GOTOinfinity/items/877fc90168d84d8d1297
<br>
[2]楽天Kドリームス<br>
https://keirin.kdreams.jp/

<br>
<font color="Green">
// Created by K.Yabu on 2022/02/22
<br>
// Copyright © 2022 K.Yabu All rights reserved.
</font>
