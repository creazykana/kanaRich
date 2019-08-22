import requests
import os
import pandas as pd
import numpy as np
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm


def checkState(req):
    if req.status_code == 200:
        pass
    else:
        print("%s web status error：%d" % (licenseNum, req.status_code))
        

def visitWeb_req(licenseNum = "92440300MA5DJJG80F"):
    targetLink = "https://www.qixin.com/search?key=%s&page=1" %licenseNum
    header = {
            "Host": "www.qixin.com",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
            "Origin": "https://www.qixin.com",
            "X-Requested-With": "XMLHttpRequest",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
            "Referer": "https://www.qixin.com"}
    #            "Referer": "https://www.qixin.com/search?key=%s&page=1" %licenseNum}
    cookie = {
            "acw_tc": "2f624a1615610986713308133e41ac79d70c7cd613037c0e50601829b241eb",
            "Hm_lvt_52d64b8d3f6d42a2e416d59635df3f71": "1561098671",
            "cookieShowLoginTip": "3",
            "Hm_lpvt_52d64b8d3f6d42a2e416d59635df3f71": "1561344724"}
    
    s = requests.get(targetLink, headers=header, verify=False)
    checkState(s)
    soup = BeautifulSoup(s.text, 'html.parser')
    return soup


def openBrowser():
    chrome_options = webdriver.ChromeOptions()
#    option.add_argument('--user-data-dir=C:/Users/hongzk/AppData/Local/Google/Chrome/User Data') #注意斜杠和反斜杠
#    chrome_options.add_argument('--disable-extensions')
#    chrome_options.add_argument('--profile-directory=Default')
#    chrome_options.add_argument("--incognito")
#    chrome_options.add_argument("--disable-plugins-discovery");
#    chrome_options.add_argument("--start-maximized")
#    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")#需提前关闭所有chrome
    driver = webdriver.Chrome(options=chrome_options) 
    return driver


def visitWeb_sln(licenseNum="92440300MA5DJJG80F", driver=None):
    if not driver:
        driver = openBrowser()
    driver.get("https://www.qixin.com/search?key=%s&page=1" % licenseNum)
    soup = BeautifulSoup(driver.page_source , 'html.parser')
    return soup


def clickCheck(driver):
    button = driver.find_element_by_class_name("btn4")
    button.click()
    

def pictureCheck(driver):
    pass



def parseHtml(soup):
    
    body = soup.body
    content = body.find("div", attrs={"class":"col-2-1"})
    if content:
        companyName = content.find("div", attrs={"class":"company-title font-18 font-f1"}).find("a").text
        detailLink = content.find("div", attrs={"class":"company-title font-18 font-f1"}).find("a").get("href")
        status = content.find("div", attrs={"class":"margin-t-0-3x"}).find("span").text
        legalPerson = content.find("div", attrs={"class":"legal-person"}).find("span").text
        registeredCapital = content.find("div", attrs={"class":"legal-person"}).find("span", attrs={"class":"margin-l-2x"}).text
        registeredTime = content.find("div", attrs={"class":"legal-person"}).find_all("span", attrs={"class":"margin-l-2x"})[1].text
        address = content.find_all("div", attrs={"class":"legal-person"})[1].text
        licenseNumSpide = content.find("span", attrs={"class":"break-word"}).text
        return [companyName, status, legalPerson, registeredCapital, registeredTime, address, licenseNumSpide, detailLink]
    else:
        return ["nodata" for i in range(8)]


if __name__=="__main__":
    orgData = pd.read_excel(r"E:/work_file/20190618_烟草局信息爬虫/data/深圳执照号名单.xlsx")
    data = orgData[["cust_code", "business_license_number_dc", "register_area"]].rename(columns={"business_license_number_dc":"licenseNum"})
    columns = ["cust_code", "licenseNum", "register_area", "companyName", "status",
               "legalPerson", "registeredCapital", "registeredTime", "address", "licenseNumSpide", "detailLink"]
    resultDf = pd.DataFrame(columns=columns)
    
    driver = openBrowser()
    tqdmObj = tqdm(range(10))
    for i in tqdmObj:
        licenseNum = data.loc[i, "licenseNum"]
        soup = visitWeb_sln(licenseNum, driver)
        infoList = parseHtml(soup)
        df = pd.DataFrame(data=[data.loc[i, :].tolist()+infoList], columns=columns)
        resultDf = resultDf.append(df)
    tqdmObj.close()
