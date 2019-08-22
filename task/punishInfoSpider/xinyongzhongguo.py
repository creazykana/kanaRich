import requests
import time
import datetime
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
import random
from selenium import webdriver
import pickle
from utils.spiderTool import proxiesIP_pool, proxiesListTest


def selenium_xyzg():
    data = pd.read_excel(r'E:/work_file/20190618_烟草局信息爬虫/data/深圳执照号名单.xlsx')
    licenseList = data["business_license_number_dc"].tolist()
    #    columns = ["license", "company", "manager", "registeredCapital", "registeredTime", "mail", "phone", "address", "registeredNum", "companyLink"]
    columns = ["license", "result"]
    resultDf = pd.DataFrame(columns=columns)

    option = webdriver.ChromeOptions()
    option.add_argument("headless")

    #    driver = webdriver.Chrome()
    driver = webdriver.Chrome(chrome_options=option)
    driver.get("http://www.gdcredit.gov.cn/infoTypeAction!xzTwoPublicListIframe_new.do?type=9")
    for i in range(31, len(licenseList)):
        licenseNum = licenseList[i]

        inputText = driver.find_element_by_class_name("search-input")
        inputText.clear()
        inputText.send_keys(licenseNum)
        #        locator = (By.CLASS_NAME, 'search-container')
        #        WebDriverWait(driver, 5, 0.5).until(EC.element_to_be_clickable(locator))#WebDriverWait(driver, 超时时长, 调用频率, 忽略异常).until(可执行方法, 超时时返回的信息)
        searchClick = True
        while searchClick:
            try:
                button = driver.find_element_by_class_name("search-container").find_element_by_tag_name("img")
                button.click()
                searchClick = False
            except:
                time.sleep(0.5)
        #        frame = driver.find_elements_by_tag_name('iframe')[0]
        #        driver.switch_to_frame(frame)
        driver.switch_to.frame(0)  # 子框架

        soup = BeautifulSoup(driver.page_source , 'html.parser')
        body = soup.body
        try:
            result = body.find("div", attrs={"class" :"list_content not_find_list"}).text
        except:
            result =[]
            content = body.find("table", attrs={"class" :"xzxx-table"})
            for record in content.find_all("tr"):
                url = "http://www.gdcredit.gov.cn " +record.find("td", attrs={"class" :"xzxx-first"}).find("a").get \
                    ("href")
                result.append(url)
        df = pd.DataFrame(data=[[licenseNum, result]], columns=columns)
        resultDf = resultDf.append(df)
        driver.switch_to.default_content()  # 退出frame

        #        licenseInfo = selenium_qcc(licenseNum)
        #        df = pd.DataFrame(data=[licenseInfo], columns=columns)
        #        resultDf = resultDf.append(df)
        #        time.sleep(0.01)
        t = i+ 1

        if t % 10 == 0:
            percent = '{:.2%}'.format(t / len(licenseList))
            print("spider %s (%d/%d) percent info successfully!" % (percent, t, len(licenseList)))
        if t % 1000 == 0:
            resultDf.to_excel(r'E:\work_file\20190618_烟草局信息爬虫\result\qccInfo_sz.xlsx', index=False)
    resultDf.to_excel(r'E:\work_file\20190618_烟草局信息爬虫\result\qccInfo_sz.xlsx', index=False)
    driver.quit()


def requests_xyzg(licenseNum, ip):
    formData = {"type": "9",
                "keyWord": licenseNum,
                "depId": "省发展改革委,省工业和信息化厅,省教育厅,省科技厅,省民族宗教委,省公安厅,省民政厅,省司法厅,省财政厅,省人力资源社会保障厅,省生态环境厅,省自然资源厅,省住房城乡建设厅,省交通运输厅,省水利厅,省农业厅,省林业局,省商务厅,省文化和旅游厅 ,人民银行广州分行,省审计厅,省气象局,省市场监管局,省药监局,省广播电视局,省新闻出版局,省体育局,省应急管理厅,省统计局,省海洋渔业局,广东证监局,省能源局,省金融办,省粮食和储备局,省外专局,省中医药局,省人防办,省国家保密局,省密码局,省档案局,省地方志办,海关总署广东分署,广东银保监局,国家税务总局广东省税务局,广州市,深圳市,珠海市,汕头市,韶关市,佛山市,江门市,湛江市,茂名市,肇庆市,惠州市,梅州市,汕尾市,河源市,阳江市,清远市,东莞市,中山市,潮州市,揭阳市,云浮市,",
                "depType": "0",
                "page": "1",
                "pageSize": "10"}
    url = "http://www.gdcredit.gov.cn/infoTypeAction!xzTwoPublicList_new.do?type=9&depType=0&refresh1561514323035"
    proxies = {"http": "http://%s" % ip, "https": "http://%s" % ip, }
    s = requests.post(url, data=formData, proxies=proxies, allow_redirects=True)
    return s


def parseHtml_xyzg(body):
    try:
        result = body.find("div", attrs={"class": "list_content not_find_list"}).text
    except:
        result = ""
        content = body.find("table", attrs={"class": "xzxx-table"})
        for record in content.find_all("tr"):
            url = "http://www.gdcredit.gov.cn" + record.find("td", attrs={"class": "xzxx-first"}).find("a").get("href")
            result = result+","+url
    df = pd.DataFrame(data=[[licenseNum, result]], columns=columns)
    return df


def multiTestIP():
    availableIP = proxiesListTest
    for i in range(3):
        df_ips = proxiesIP_pool(requests_xyzg, availableIP)
        availableIP = (df_ips[df_ips["timeUse"] < 2])["IP"].tolist()
        print("test %s times"%str(i+1))
    return availableIP


def searchInfo(licenseList, availableIP):
    os.chdir(r'E:\work_file\20190618_烟草局信息爬虫\result')
    resultFileName = '行政处罚查询结果_from信用中国_append.xlsx'
    columns = ["license", "result"]
    resultDf = pd.DataFrame(columns=columns)
    errorTime = 0
    errorInfo = pd.DataFrame(columns=["id", "license"])
    for i in range(len(licenseList)):
        time.sleep(0.1)
        licenseNum = licenseList[i]
        randomIndex = random.randint(0, len(availableIP) - 1)
        ip = availableIP[randomIndex]
        try:
            s = requests_xyzg(licenseNum, ip)
            soup = BeautifulSoup(s.text, 'html.parser')
            body = soup.body
            df = parseHtml_xyzg(body)
            resultDf = resultDf.append(df)
        except:
            errorTime += 1
            print("Error:licenseNum %s can't get the info!(accumulate %d)"%(licenseNum, errorTime))
            errorInfo = errorInfo.append(pd.DataFrame(data=[[i, licenseNum]], columns=["id", "license"]))
            if errorTime > 50:
                errorInfo.to_excel('errorInfo.xlsx', index=False)
                resultDf.to_excel(resultFileName, index=False)
                break
        t = i + 1
        if t % 10 == 0:
            percent = '{:.2%}'.format(t / len(licenseList))
            print("spider %s (%d/%d) percent info successfully!" % (percent, t, len(licenseList)))
        if t % 1000 == 0:
            resultDf.to_excel(resultFileName, index=False)
    resultDf.to_excel(resultFileName, index=False)
    return resultDf



def getPunishInfo(url, ip):
    proxies = {"http": "http://%s" % ip, "https": "http://%s" % ip, }
    s = requests.get(url, proxies=proxies, verify=False)
    soup = BeautifulSoup(s.text, 'html.parser')
    body = soup.body
    infoTable = body.find("div", attrs={"class":"content infoType-content"}).find("table")
    records_firts = infoTable.find_all("tr", attrs={"class": "first"})
    records_seconde = infoTable.find_all("tr", attrs={"class": "seconde"})

    punishDecideNo = records_seconde[0].find("td", attrs={"class": "value"}).text
    punishName = records_firts[0].find("td", attrs={"class": "value"}).text
    punishType_1 = records_seconde[1].find("td", attrs={"class": "value"}).text
    punishType_2 = records_firts[1].find("td", attrs={"class": "value"}).text
    punishReason = records_seconde[2].find("td", attrs={"class": "value"}).text
    accordingRule = records_firts[2].find("td", attrs={"class": "value"}).text
    punishObj = records_firts[3].find("td", attrs={"class": "value"}).text
    societyCreditCode = records_seconde[3].find("tr", attrs={"class": "value"}).find_all("td")[0].text
    organizationCode = records_seconde[3].find("tr", attrs={"class": "value"}).find_all("td")[1].text
    ICregisterCode = records_seconde[3].find("tr", attrs={"class": "value"}).find_all("td")[2].text
    taxRegisterCode = records_seconde[3].find("tr", attrs={"class": "value"}).find_all("td")[3].text
    IDcode = records_seconde[3].find("tr", attrs={"class": "value"}).find_all("td")[4].text
    legalPersonName = records_firts[4].find("td", attrs={"class": "value"}).text
    punishDecideDate = records_seconde[4].find("td", attrs={"class": "value"}).text
    punishOrgan = records_firts[5].find("td", attrs={"class": "value"}).text
    localCode = records_seconde[5].find("td", attrs={"class": "value"}).text
    status = records_firts[6].find("td", attrs={"class": "value"}).text
    updateTime = records_seconde[6].find("td", attrs={"class": "value"}).text
    remark = records_firts[7].find("td", attrs={"class": "value"}).text

    columns = ["punishDecideNo", "punishName", "punishType_1", "punishType_2", "punishReason", "accordingRule", "punishObj",
              "societyCreditCode", "organizationCode", "ICregisterCode", "taxRegisterCode", "IDcode", "legalPersonName",
              "punishDecideDate", "punishOrgan", "localCode", "status", "updateTime", "remark"]
    result = [punishDecideNo, punishName, punishType_1, punishType_2, punishReason, accordingRule, punishObj,
              societyCreditCode, organizationCode, ICregisterCode, taxRegisterCode, IDcode, legalPersonName,
              punishDecideDate, punishOrgan, localCode, status, updateTime, remark]
    result = [re.sub("\s", "", i) for i in result]
    df = pd.DataFrame(data=[result], columns=columns)
    return df


if __name__ == "__main__":
    basicInfoResult = pd.read_excel(r'E:/work_file/20190618_烟草局信息爬虫/result/行政处罚查询结果_from信用中国.xlsx')
    columns = ["punishDecideNo", "punishName", "punishType_1", "punishType_2", "punishReason", "accordingRule", "punishObj",
              "societyCreditCode", "organizationCode", "ICregisterCode", "taxRegisterCode", "IDcode", "legalPersonName",
              "punishDecideDate", "punishOrgan", "localCode", "status", "updateTime", "remark"]
    punishInfo = pd.DataFrame(columns=columns)
    punishUrl = basicInfoResult[basicInfoResult["result"]!="找不到和您的查询相符的记录。"]["result"].tolist()
    for url_str in punishUrl:
        split_url = url_str.split(",")
        for url in split_url:
            if len(url)==0:
                pass
            else:
                randomIndex = random.randint(0, len(availableIP) - 1)
                ip = availableIP[randomIndex]
                df = getPunishInfo(url, ip)
                punishInfo = punishInfo.append(df)
    punishInfo.to_excel('punishInfo_gd.xlsx', index=False)

