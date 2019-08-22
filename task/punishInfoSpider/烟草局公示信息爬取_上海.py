# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:32:02 2019

@author: hongzk
"""
import requests
import os
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np


targetLink = "https://www.sh.tobacco.com.cn/qygs/monopolyManage/8a8a8b995d0af15a015d0b9f020d0350.htm?curNav=4&v=18121801"
#直接获取网页html无法得到目标数据，目标网页的表格数据应是javascript动态加载,需要找到真正的数据来源网址
headers = {
        'Accept': 'application/json, text/javascript, */*; q=0.01',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Referer': 'https://www.sh.tobacco.com.cn/qygs/monopolyManage/8a8a8b995d0af15a015d0b9f020d0350.htm?curNav=4&v=18121801',
        'Origin': 'https://www.sh.tobacco.com.cn',
        'Host': 'www.sh.tobacco.com.cn',
        'Connection': 'keep-alive'
        }

cookies = {
        'JSESSIONID':'GbdpOJv0Jr1A01nOA7L2Pw08ZoJQdQjuSWkwB4OHd4M9gJflYE0g!1274755800',
        'tipbox':'close'
        }

s = requests.get(targetLink, headers=headers, cookies=cookies, verify=False)
print("网页获取状态：%d"%s.status_code)
soup = BeautifulSoup(s.text, 'html.parser')
body = soup.body
infoTable = body.find_all("div", attrs={"class":"table"})


# =============================================================================
# 
# =============================================================================
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import re
from PIL import Image
import pandas as pd
import os
import time   
os.chdir(r'E:\work_file\20190618_烟草局信息爬虫\result')

targetLink = "https://www.sh.tobacco.com.cn/qygs/monopolyManage/8a8a8b995d0af15a015d0b9f020d0350.htm?curNav=4&v=18121801"
dataColumns = ['办案区局', '立案时间', '立案号', '当事人', '案件名称', '业务时间',
               '处罚决定', '处罚决定书号', '社会信用代码', '处罚事由', '处罚依据', '处罚日期']
resultData = pd.DataFrame(columns=dataColumns)

option = webdriver.ChromeOptions()
option.add_argument("headless")
driver = webdriver.Chrome()
driver.get(targetLink)
soup = BeautifulSoup(driver.page_source , 'html.parser')
body = soup.body
pages = int(re.sub(".*,", "", body.find("a", attrs={"class":"nextPage"}).get("onclick")).replace(")", ""))

for i in range(pages):
    soup = BeautifulSoup(driver.page_source , 'html.parser')
    body = soup.body
    infoTable = body.find("div", attrs={"class":"table"})
    infoLists = infoTable.find_all("tr", attrs={"id":"xkgs"})
    for record in infoLists:
        allInfo = record.find_all("td")
        recordInfo = []
        for i in range(len(allInfo)-1):
            info = allInfo[i].text
            recordInfo.append(info)
        df = pd.DataFrame(data=[recordInfo], columns=dataColumns)
        resultData = resultData.append(df)
    
    page = re.sub(".*\(", "", body.find("a", attrs={"class":"nextPage"}).get("onclick")).replace(")", "").replace(",", "-")
    print("-->> %s page info had got "%page)
    time.sleep(0.5)
    try:
        elem = driver.find_element_by_class_name("nextPage")
        elem.click()
    except:
        print("---->>>>can't find next page!!!")
resultData.to_excel('custPunishInfo_sh.xlsx', index=False)




