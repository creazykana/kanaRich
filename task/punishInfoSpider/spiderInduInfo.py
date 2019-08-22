# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:46:50 2019

@author: hongzk
"""

import requests
import time
import datetime
from bs4 import BeautifulSoup
import pandas as pd
import re
import random
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from io import BytesIO
from PIL import Image




def getSimpleInfo_qxb(licenseNum):
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
    s = requests.get(targetLink, headers=header, cookies=cookie, verify=False)
    print("网页获取状态：%d"%s.status_code)
    soup = BeautifulSoup(s.text, 'html.parser')
    body = soup.body
    content = body.find("div", attrs={"class":"col-2-1"})
    companyName = content.find("div", attrs={"class":"company-title font-18 font-f1"}).find("a").text
    detailLink = content.find("div", attrs={"class":"company-title font-18 font-f1"}).find("a").get("href")
    status = content.find("div", attrs={"class":"margin-t-0-3x"}).find("span").text
    legalPerson = content.find("div", attrs={"class":"legal-person"}).find("span").text
    registeredCapital = content.find("div", attrs={"class":"legal-person"}).find("span", attrs={"class":"margin-l-2x"}).text
    registeredTime = content.find("div", attrs={"class":"legal-person"}).find_all("span", attrs={"class":"margin-l-2x"})[1].text
    address = content.find_all("div", attrs={"class":"legal-person"})[1].text
    licenseNumSpide = content.find("span", attrs={"class":"break-word"}).text
    return [companyName, status, legalPerson, registeredCapital, registeredTime, address, licenseNumSpide, detailLink]


def cookiePool(randomIndex=-1):
    cookies = [{"UM_distinctid":"16a0174b1089f4-0c5bfd48c2bcb8-7a1437-15f900-16a0174b109918",
                "zg_did":"%7B%22did%22%3A%20%2216a0174b1e9301-019f849634998-7a1437-15f900-16a0174b1ea96f%22%7D",
                "_uab_collina":"155480258675068326260235",
                "acw_tc":"9dff8bc915586980291772462ea05d34b9622c9d0e21e03c5e2aac176f",
                "QCCSESSID":"mti0rh0mc04vdhifmbvkeuo706",
                "Hm_lvt_3456bee468c83cc63fb5147f119f1075":"1559030143,1559209530,1560924760,1560926068",
                "CNZZDATA1254842228":"1206815491-1554800990-https%253A%252F%252Fwww.google.com%252F%7C1561354286",
                "hasShow":"1",
                "zg_de1d1a35bfa24ce29bbf2c7eb17e6c4f":"%7B%22sid%22%3A%201561358583849%2C%22updated%22%3A%201561358862666%2C%22info%22%3A%201560924759421%2C%22superProperty%22%3A%20%22%7B%7D%22%2C%22platform%22%3A%20%22%7B%7D%22%2C%22utm%22%3A%20%22%7B%7D%22%2C%22referrerDomain%22%3A%20%22%22%2C%22cuid%22%3A%20%22eddfdc22d7bf711a0715713558c4ad1d%22%7D",
                "Hm_lpvt_3456bee468c83cc63fb5147f119f1075":"1561361569" #登陆信息
                },
                {"UM_distinctid":"169fc264c930-0fd05565ecf791-7c103c49-1fa400-169fc264c94949",
                "zg_did":"%7B%22did%22%3A%20%22169fc2650db36a-097c9e44a2b14e-7c103c49-1fa400-169fc2650dc31%22%7D",
                "acw_tc":"9dff8bc915613674343156391e4e58bca9f8bb16d36e8a11fd943e3fdc",
                "QCCSESSID":"edbaeu38efb4a1drkabkjf04i3",
                "CNZZDATA1254842228":"1110940098-1554709067-https%253A%252F%252Fwww.google.com%252F%7C1561365086",
                "Hm_lvt_3456bee468c83cc63fb5147f119f1075":"1561367435",
                "Hm_lpvt_3456bee468c83cc63fb5147f119f1075":"1561367664",
                "zg_de1d1a35bfa24ce29bbf2c7eb17e6c4f":"%7B%22sid%22%3A%201561367433776%2C%22updated%22%3A%201561367697296%2C%22info%22%3A%201561367433780%2C%22superProperty%22%3A%20%22%7B%7D%22%2C%22platform%22%3A%20%22%7B%7D%22%2C%22utm%22%3A%20%22%7B%7D%22%2C%22referrerDomain%22%3A%20%22%22%2C%22cuid%22%3A%20%2279f21927cb0e1c654506d73f05c81186%22%7D"
                }]
    if randomIndex<0:
        randomIndex = random.randint(0, len(cookies)-1)
    return cookies[randomIndex]

def requests_qcc(licenseNum):
    requests.packages.urllib3.disable_warnings()#解决requests因忽略证书验证而报错
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
               "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
               "Referer": "https://www.qichacha.com/",
               "Host": "www.qichacha.com"}
    cookies = cookiePool(0)
#    cookies = {"UM_distinctid":"169fc264c930-0fd05565ecf791-7c103c49-1fa400-169fc264c94949",
#                "zg_did":"%7B%22did%22%3A%20%22169fc2650db36a-097c9e44a2b14e-7c103c49-1fa400-169fc2650dc31%22%7D",
#                "acw_tc":"9dff8bc915613674343156391e4e58bca9f8bb16d36e8a11fd943e3fdc",
#                "QCCSESSID":"edbaeu38efb4a1drkabkjf04i3",
#                "CNZZDATA1254842228":"1110940098-1554709067-https%253A%252F%252Fwww.google.com%252F%7C1561365086",
#                "Hm_lvt_3456bee468c83cc63fb5147f119f1075":"1561367435",
#                "Hm_lpvt_3456bee468c83cc63fb5147f119f1075":"1561367664",
#                "zg_de1d1a35bfa24ce29bbf2c7eb17e6c4f":"%7B%22sid%22%3A%201561367433776%2C%22updated%22%3A%201561367697296%2C%22info%22%3A%201561367433780%2C%22superProperty%22%3A%20%22%7B%7D%22%2C%22platform%22%3A%20%22%7B%7D%22%2C%22utm%22%3A%20%22%7B%7D%22%2C%22referrerDomain%22%3A%20%22%22%2C%22cuid%22%3A%20%2279f21927cb0e1c654506d73f05c81186%22%7D"
#                }
    searchDict = {"key":licenseNum}
#    randomIndex = random.randint(0, len(availableIP))
#    ip = availableIP[randomIndex]    
#    proxies = { "http": "http://%s"%ip, "https": "http://%s"%ip, }  #, proxies=proxies
    s = requests.get('https://www.qichacha.com/search', params=searchDict, headers=headers, cookies=cookies, verify=False)
    print("statusOcode:%s"%str(s.status_code))  
    soup = BeautifulSoup(s.text, 'html.parser')
    body = soup.body
    return body

def selenium_qcc(licenseNum):
#    option = webdriver.ChromeOptions()
#    option.add_argument("headless")
#    driver = webdriver.Chrome()
    
#    selenium_cookies = {
#            "name":"QCCSESSID",
#            "value":"mti0rh0mc04vdhifmbvkeuo706"
#            }
#    driver.get("https://www.qichacha.com/search?key=%s"%licenseNum)
#    driver.delete_cookie('QCCSESSID')
#    driver.add_cookie(cookie_dict=selenium_cookies)
    driver.get("https://www.qichacha.com/search?key=%s"%licenseNum)
    soup = BeautifulSoup(driver.page_source , 'html.parser')
    body = soup.body
    if body.find("div", attrs={"class":"text-center regTab m-t-xl"}):
        machineCheck_qcc(driver)
        while body.find("span", attrs={"class":"nc-lang-cnt"}):
            refresh = driver.find_element_by_class_name("nc-lang-cnt").find_element_by_tag_name("a")
            refresh.click()
            time.sleep(2)
            machineCheck_qcc(driver)
    return parseHtml_qcc(body)


def parseHtml_qcc(body):
    if body.find("section", attrs={"class":"panel panel-default"}):
        print("%s is not data" %licenseNum)
        return [licenseNum, "", "", "", "", "", "", "", "", ""]

    search_result = body.find("table", attrs={"class":"m_srchList"})    
    content = search_result.find_all("td")[2]

    companyName = content.find("a").text
    companyLink = content.find("a").get("href")    
    manager = content.find_all("p")[0].find("a").text
    registeredCapital = content.find_all("p")[0].find_all("span")[0].text
    registeredTime = content.find_all("p")[0].find_all("span")[1].text
    mail = re.sub("\s", "", content.find_all("p")[1].text)
    phone = content.find_all("p")[1].find("span").text
    address = re.sub("\s", "", content.find_all("p")[2].text)
    registeredNum = re.sub("\s", "", content.find_all("p")[3].text)
    return [licenseNum, companyName, manager, registeredCapital, registeredTime, mail, phone, address, registeredNum, companyLink]


def get_track(distance):
    """
    根据偏移量获取移动轨迹
    :param distance: 偏移量
    :return: 移动轨迹
    """
    # 移动轨迹
    track = []
    # 当前位移
    current = 0
    # 减速阈值
    mid = distance * 4 / 5
    # 计算间隔
    t = 0.2
    # 初速度
    v = 0
    
    while current < distance:
        if current < mid:
            # 加速度为正2
            a = 1.5
        else:
            # 加速度为负3
            a = -2.5
        # 初速度v0
        v0 = v
        # 当前速度v = v0 + at
        v = v0 + a * t
        # 移动距离x = v0t + 1/2 * a * t^2
        move = v0 * t + 1 / 2 * a * t * t
        # 当前位移
        current += move
        # 加入轨迹
        track.append(round(move))
    return track

def machineCheck_qcc(driver):
    wait = WebDriverWait(driver, 20)
    img = driver.find_element_by_id("nc_1__scale_text")
    time.sleep(2)
    location = img.location
    size = img.size
    top, bottom, left, right = location['y'], location['y'] + size['height'], location['x'], location['x'] + size[
        'width']

    screenshot = driver.get_screenshot_as_png()
    screenshot = Image.open(BytesIO(screenshot))
    slider = driver.find_element_by_id("nc_1_n1z") #找到滑块
    captcha = screenshot.crop((left, top, right, bottom))
    captcha.save("captcha.png")
    gap = right - left
    track = get_track(gap)
    ActionChains(driver).click_and_hold(slider).perform()
    for x in track:
        ActionChains(driver).move_by_offset(xoffset=x, yoffset=0).perform()
    time.sleep(1)
    ActionChains(driver).release().perform()





#2.8%   18:00
