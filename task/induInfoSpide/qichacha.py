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
    driver.get("https://www.qichacha.com/search?key=%s" % licenseNum)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    body = soup.body
    if body.find("div", attrs={"class": "text-center regTab m-t-xl"}):
        machineCheck_qcc(driver)
        while body.find("span", attrs={"class": "nc-lang-cnt"}):
            refresh = driver.find_element_by_class_name("nc-lang-cnt").find_element_by_tag_name("a")
            refresh.click()
            time.sleep(2)
            machineCheck_qcc(driver)
    return parseHtml_qcc(body)


def parseHtml_qcc(body):
    if body.find("section", attrs={"class": "panel panel-default"}):
        print("%s is not data" % licenseNum)
        return [licenseNum, "", "", "", "", "", "", "", "", ""]

    search_result = body.find("table", attrs={"class": "m_srchList"})
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
    return [licenseNum, companyName, manager, registeredCapital, registeredTime, mail, phone, address, registeredNum,
            companyLink]





def machineCheck_qcc(driver):
    wait = WebDriverWait(driver, 20)
    img = driver.find_element_by_id("dom_id_one")
    time.sleep(2)
    location = img.location
    size = img.size
    top, bottom, left, right = location['y'], location['y'] + size['height'], location['x'], location['x'] + size[
        'width']

    screenshot = driver.get_screenshot_as_png()
    screenshot = Image.open(BytesIO(screenshot))
    slider = driver.find_element_by_id("nc_1_n1z")  # 找到滑块 id  nc_2_n1z    class nc_iconfont btn_slide
    captcha = screenshot.crop((left, top, right, bottom))
    captcha.save("captcha.png")
    gap = right - left
    track = get_track(gap)
    ActionChains(driver).click_and_hold(slider).perform()
    for x in track:
        ActionChains(driver).move_by_offset(xoffset=x, yoffset=0).perform()
    time.sleep(1)
    ActionChains(driver).release().perform()



def simulateSignIn(accountList): #selenium打开的界面进行的滑动验证都失败(手动也一样)，问题未解决！
    account = accountList[0]
    password = accountList[1]

    chrome_options = webdriver.ChromeOptions()
#    option.add_argument('--user-data-dir=C:/Users/hongzk/AppData/Local/Google/Chrome/User Data') #注意斜杠和反斜杠
#    chrome_options.add_argument('--disable-extensions')
#    chrome_options.add_argument('--profile-directory=Default')
#    chrome_options.add_argument("--incognito")
#    chrome_options.add_argument("--disable-plugins-discovery");
#    chrome_options.add_argument("--start-maximized")
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    driver = webdriver.Chrome(options=chrome_options) #需提前关闭所有chrome
#    driver.delete_all_cookies()
#    driver.set_window_size(800,800)
#    driver.set_window_position(0,0)
    print('arguments done')
    driver.get("https://www.qichacha.com/user_login?back=%2F")
    driver.find_element_by_id("normalLogin").click()
    inputAccount = driver.find_element_by_id("nameNormal")
    inputPassword = driver.find_element_by_id("pwdNormal")
    inputAccount.clear()
    inputAccount.send_keys(account)
    inputPassword.clear()
    inputPassword.send_keys(password)
    machineCheck_qcc(driver)
    driver.close()

    # chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")
    # chrome_driver = "C:\Program Files (x86)\Google\Chrome\Application\chromedriver.exe"
    # driver = webdriver.Chrome(chrome_options=chrome_options)

# 首先cmd 运行命令chrome.exe --remote-debugging-port=9222 打开一个浏览器，
# 然后py代码里
# 添加一个这个Options。其它的代码不变

if __name__ == "__main__":
    accountPool = [["18408291935", "iqyhygxcom"],
                   ["13333045057", "z0000oooo"],
                   ["18974931420", "12345678"],
                   ["17115615391", "qqq123456"],
                   ["18819030141", "666888"]
                   ]
    accountList = accountPool[0]
