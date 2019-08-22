# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:47:55 2019

@author: hongzk
"""
import requests
import os
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import json
import pickle
from collections import Counter

def openURL(url, searchDict={}, headers={}, cookies={}, verify=False):
    if len(searchDict)==0:
        s = requests.get(url, headers=headers, cookies=cookies, verify=verify)
    else:
        s = requests.get(url, headers=headers, params=searchDict, cookies=cookies, verify=verify)
    print("网页获取状态：%d"%s.status_code)
    soup = BeautifulSoup(s.text, 'html.parser')
    body = soup.body
    return body

def getLocalTobaccoLink():
    result = pd.DataFrame(columns=["province", "web_link", "web_status"])
    tobacco_china = "http://www.tobacco.gov.cn/html/"
    body = openURL(tobacco_china)
    content = body.find_all("div", attrs={"class":"fang_line"})[2]
    linkDict = {}
    for html in content.find_all("td"):
        province = re.sub("\s", "", html.find("a").text.encode('ISO-8859-1').decode('gbk'))
        url = html.find("a").get("href")
        t = requests.get(url, verify=False) #应添加响应时间限制
        status = t.status_code
        df = pd.DataFrame(data=[[province, url, status]], columns=["province", "web_link", "web_status"])
        result = result.append(df)
    return result

def getBrandInfo():
    url = "http://www.fjycw.com/yp_data/yp_data.ashx"
    header = {
        'Host':'www.fjycw.com',
        'Accept':'application/json, text/javascript, */*; q=0.01',
        'X-Requested-With':'XMLHttpRequest',
        'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
        'Referer':'http://www.fjycw.com/yp_default.aspx',
        'Accept-Encoding':'gzip, deflate',
        'Accept-Language':'zh-CN,zh;q=0.9',
        'Referer':'http://www.fjycw.com/yp_default.aspx'
        }
    cookie = {
        'ASP.NET_SessionId':'mui4du34u02ou055lzd1jcbf',
        'UM_distinctid':'16b72d947c37c8-0897189d8c526d-e353165-15f900-16b72d947c46fb',
        'Hm_lvt_fddf33d052f5f53d3b4b95fe92b20b65':'1560999971', 
        'CNZZDATA1258291724':'1591766742-1560996343-http%253A%252F%252Ffj.tobacco.gov.cn%252F%7C1561006311', 
        'CNZZDATA1770494':'cnzz_eid%3D83673880-1560999314-http%253A%252F%252Ffj.tobacco.gov.cn%252F%26ntime%3D1561010309', 
        'Hm_lpvt_fddf33d052f5f53d3b4b95fe92b20b65':'1561011232'}
    data = {
            'type':'news',
            'address':'',
            'pp':'',
            'money':'',
            'jy':'',
            'wd':''
            }
    columns = ['id', 'name', 'pp', 'ower', 'comp', 'address', 'class', 'guige', 'sale',
               'saleDW', 'saleDWNum', 'jiaoyou', 'niguding', 'o1', 'sma', 'bma', 'pinglei',
               'addtime', 'order', 'image', 'smallimage', 'memo', 'hitTime', 'hitDay',
               'hits', 'en', 'en1', 'isSale', 'md', 'area', 'baozhuan', '3d', 'ens1',
               'image1', 'className', 'classimage']
    record = requests.post(url, data=data, headers=header, cookies=cookie, allow_redirects=True).json()
    recordList = json.loads(json.dumps(record))
    result = pd.DataFrame(columns=columns)
    for data in recordList:
        datas = list(data.values())
        df = pd.DataFrame(data=[datas], columns=columns)
        result = result.append(df)
    return result
        
def punishInfo_hn():
    result = pd.DataFrame(columns=["name", "title", "case", "date"])
    url = "http://222.240.173.99:9020/index.do?method=doXzcfList&orgid=01111430001&bzid=01"
    body = openURL(url)
    infoTable = body.find("ul", attrs={"id":"www_zzjs_net"}).find_all("li")
    for record in infoTable:
        content = record.find_all("td")
        name = content[0].text
        title = content[1].text
        case = content[2].text
        date = content[3].text
        df = pd.DataFrame(data=[[name, title, case, date]], columns=["name", "title", "case", "date"])
        result = result.append(df)
    return result        

def punishInfo_cq():
    def getLinks(url):
        body = openURL(url)
        contents = body.find("div", attrs={"class":"artlist-list"})
        urlTitle = []
        for content in contents.find_all("li"):
            try:
                html = content.find("a").get("href")
                title = content.find("a").text
                urlTitle.append([title, html])
            except:
                pass
        return urlTitle
    #先获取所有的url，再解析url里面的处罚信息
    urlDf = pd.DataFrame(columns = ["title", "url"])
    for i in range(1,6):
        if i==1:
            IllegalListLink = "http://cq.tobacco.gov.cn/c/22"
        else:
            IllegalListLink = "http://cq.tobacco.gov.cn/c/22/%d"%i
        urlTitle = getLinks(IllegalListLink)
        df = pd.DataFrame(data=urlTitle, columns=["title", "url"])
        urlDf = urlDf.append(df)
        urlDf = urlDf.drop_duplicates().reset_index(drop=True)

    print("--->>>had got all illegal info link")

    dfDict = {}
    for j in range(len(urlDf)):
        url = 'http://cq.tobacco.gov.cn'+urlDf.loc[j, "url"]
        df = pd.read_html(url)[0]
#        df.to_excel(r"E:\work_file\20190618_烟草局信息爬虫\result\重庆违法信息\%d.xlsx"%j, index=False)
        df_feature = np.array(df[1:2]).tolist()
        if len(dfDict)==0:
            dfDict[0] = {"feature":df_feature, "dataframe":df}
        else:
            addData = True
            for i in range(len(dfDict)):
                feature = dfDict[i]["feature"]
                data = dfDict[i]["dataframe"]
                if df_feature==feature:
                    data = data.append(df[2:])
                    dfDict[i]["dataframe"] = data
                    addData = False
                else:
                    pass
            if addData:
                maxFeatureID = len(dfDict)
                dfDict[maxFeatureID] = {"feature":df_feature, "dataframe":df}
    for k in range(len(dfDict)):
        dfDict[k]["dataframe"].to_excel(r"E:\work_file\20190618_烟草局信息爬虫\result\重庆违法信息\信息集合%d.xlsx"%k, index=False)

        
if __name__ == "__main__":
    test = getLocalTobaccoLink()
    test.to_excel(r'E:\work_file\20190618_烟草局信息爬虫\result\province_company_url.xlsx', index=False)
    hunan = punishInfo_hn()
    hunan.to_excel(r'E:\work_file\20190618_烟草局信息爬虫\result\处罚信息_湖南.xlsx', index=False)
    brand_info = getBrandInfo()
    brand_info.to_excel(r'E:\work_file\20190618_烟草局信息爬虫\result\烟草信息_from_fj.xlsx', index=False)
    cq_link, punishInfo_cq = punishInfo_cq()
    fw = open(r'E:\work_file\20190618_烟草局信息爬虫\result\违法信息_湖南.pkl', 'wb')
    pickle.dump(punishInfo_cq, fw)
    fw.close()






