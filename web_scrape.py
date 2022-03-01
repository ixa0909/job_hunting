# -*- coding: utf-8 -*-
from bs4 import BeautifulSoup as bs4
import requests
import re
import time

#URL作成
#数年分取れるように拡張
def createURL(year,month, day):
    url = 'https://keirin.kdreams.jp/racecard/20'+str(year).zfill(2)+'/' + str(month).zfill(2) + '/' + str(day).zfill(2) + '/'
    return url
#データの期間設定
#数年分取れるように拡張 試験用2019/04/01のデータ取得
for term in range(20,21):
    seedURLs = [ createURL(y, m, d) for y in range(term, term+1, 1) for m in range(4, 5) for d in range(1,2)]


#各レース出走表掲載WebページのURL一覧からURL(返り値)を取得
#間違っているところを修正・改善
    def get_race_urls(sourceURLs):
        #URLを格納するための辞書を定義
        race_urls = {}
        for sourceURL in sourceURLs:
            try:
                #リクエストを作成
                req = requests.get(sourceURL)
                #htmlデータを取得、parserは読み取り形式
                soup = bs4(req.content, 'html.parser')
                #1秒待機
                time.sleep(1)
                #一覧と各ラウンドの出走表WebページのURLを取得
                race_htmls = soup.find_all('a', class_='JS_POST_THROW')
                for race_html in race_htmls:
                    url = race_html.get('href')
                #"一覧"はいらないのでURLの形式を基に除外
                    if 'racedetail' in url:
                        race_id = re.sub(r'\D', '', url)
                        race_urls[race_id] = url
            #---------
            #修正
            except:
                continue
        return race_urls

    race_urls = get_race_urls(seedURLs)


    #各レース出走表Webページからデータ取得
    import traceback
    import pandas as pd

    main_colum = ['予想', '好気合', '総評', '枠番', '車番', '選手名府県/年齢/期別', '級班', '脚質', 'ギヤ倍数', '競走得点', 'S','B','逃','捲','差','マ','1着', '2着', '3着', '着外','勝率','2連対率','3連対率']
    result_colum = ['予想', '着順', '車番', '選手名', '着差', '上り', '決まり手', 'S/B', '勝敗因']

    def scrape(race_urls, pre_race_results={}):
        race_results = pre_race_results
        
        for race_id, url in race_urls.items():
            #race_idが重なった時の対処
            if race_id in race_results.keys():
                continue

            try:
                #htmlを取得
                req=requests.get(url)
                soup = bs4(req.content, 'html.parser')

                #その日の天気データを取得
                #試し
                """
                s=str(soup.find_all("p",class_="weather"))
                print(s[s.find("雨"):])
                """

                #html内の表データをDataFrameのリストで取得
                main = pd.read_html(url)

                #出走表の表だけを取得
                #修正　main[4]だと取得部分が違うのでmain[0]に変更
                df = main[0][:-1]
                df.columns = main_colum

                #レース結果の表を取得
                result_table = main[-2]
                result_table.columns = result_colum
                df_result = result_table.loc[ : , ['着順', '車番']]

                #str型に変換
                df = df.astype(str)
                df_result = df_result.astype(str)

                #出走表データとレース結果データを一つにまとめる
                df = pd.merge(df_result, df, on='車番', how='left')
                race_results[race_id] = df

                #1秒待機
                time.sleep(1)

            #各例外に対応してメッセージを出力
            except IndexError:
                print('IndexError: {}', url)
                continue
            except KeyError:
                print('keyerror: {}', url)
                continue
            except ValueError:
                print("ValueError: {}", url)
                continue
            except :
                traceback.print_exc()
                break
        return race_results

    results={}
    results = scrape(race_urls, results)

    #各レースデータフレイムの行名をレースIDに変更
    for key in results.keys():
        results[key].index = [key]*len(results[key])

    #-----------------
    #レース出場者が9人に未たない時に空データを加える
    for key in results.keys():
        len_frame=len(results[key].index)
        if len_frame<9:
            vacant_pd=pd.DataFrame(index=range(9-len_frame))
            results[key]=results[key].append(vacant_pd,ignore_index=True)

    #データフレイム結合
    race_results = pd.concat([results[key] for key in results.keys()], sort=False)

    #file保存 ある年の「a=前半」 「b=後半」を意味し半年ずつ取得を想定
    race_results.to_pickle("data\\race_data_20"+str(term)+"a.pkl")
