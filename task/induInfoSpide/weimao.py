from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import os
import selenium
from selenium import webdriver
from fontTools.ttLib import TTFont
from tqdm import tqdm
import random
#from utils.spiderTool import proxiesIP_pool

# =============================================================================
# 目前反爬手段：自定义字体反爬(为解决)
# =============================================================================
def checkState(req):
    if req.status_code == 200:
        pass
    else:
        print("%s web status error：%d" % (licenseNum, req.status_code))

availableIP = """219.145.117.69:4251
111.72.153.47:4251
113.64.197.46:4287
14.106.107.189:4237
113.120.63.47:4264
112.87.12.245:4207
106.57.23.50:4228
119.114.74.97:4252
115.213.98.14:4286
180.126.165.118:4276""".split("\n")

def visitWeb_req(licenseNum = "92440300MA5DJJG80F", availableIP=availableIP):
    targetLink = "https://www.weimao.com/search?key=%s&type=enterprise&filter=" %licenseNum
    header = {
            "Host": "www.weimao.com",
            "Upgrade-Insecure-Requests": "1",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36",
            "Referer": "https://www.weimao.com/"}
    cookies = {"Hm_lvt_c99fc61dc621e6b02c50ffce880b90b7":"1561690741,1561948338,1562057669",
            "weimao_session":"s%3AXS3Ns1gKt2QtNu-8LonUlmMsQM8cWgTe.PB6nio4o%2Bagood%2FSELyw5Zm7vMUYtNHS%2FjreYqxKJQQ",
            "Hm_lpvt_c99fc61dc621e6b02c50ffce880b90b7":"1562142066"}
    
    randomIndex = random.randint(0, len(availableIP) - 1)
    ip = availableIP[randomIndex]

    proxies = { "http": "http://%s"%ip, "https": "http://%s"%ip, }
    req = requests.get(targetLink, proxies=proxies, headers=header, cookies=cookies, verify=False, timeout=3)
    checkState(req)
    return req



def parseHtml(obj):
    columns = ["companyName", "legalPerson", "foundTime", "registerCapital", "spideLicense", "companyInfoLink"]
    resultDf = pd.DataFrame(columns=columns)
    if isinstance(obj, selenium.webdriver.chrome.webdriver.WebDriver):
        soup = BeautifulSoup(obj.page_source , 'html.parser')
    elif isinstance(obj, requests.models.Response):
        soup = BeautifulSoup(obj.text, 'html.parser')
    body = soup.body
    content = body.find("div", attrs={"class":"main-contain"})
    companyList = content.find_all("div", attrs={"class":"company-li"})
    if len(companyList)>0:
        for company in companyList:
            detailInfo = company.find("div", attrs={"class":"search-left-content"})
            basicInfo = detailInfo.find("div", attrs={"class":"search-company-info"}).find_all("span", attrs={"class":"info-li"})

            status = company.find("div", attrs={"class":"status"}).text
            
            companyName = detailInfo.find("div", attrs={"class":"wraper-liat-a"}).find("a").text  
            companyInfoLink = detailInfo.find("div", attrs={"class":"wraper-liat-a"}).find("a").get("href")
            legalPerson = basicInfo[0].find_all("span")[1].text  
            foundTime = basicInfo[1].find_all("span")[1].text #全角转半角
            registerCapital = basicInfo[2].find_all("span")[1].text
            spideLicense = detailInfo.find("div", attrs={"class":"company-adress"}).find("em").text
            
            result = [companyName, legalPerson, foundTime, registerCapital, spideLicense, companyInfoLink]
            df = pd.DataFrame(data=[result], columns=columns)
            resultDf = resultDf.append(df)
    else:
        pass
    return resultDf
            



def getDetailInfoLink():
    orgData = pd.read_excel(r"E:/work_file/20190618_烟草局信息爬虫/data/深圳执照号名单.xlsx")
    data = orgData[["cust_code", "business_license_number_dc", "register_area"]].rename(columns={"business_license_number_dc":"licenseNum"})
#    licenseNum = "92440300MA5DJJG80F"
   
    columns = ["companyName", "legalPerson", "foundTime", "registerCapital", "spideLicense", "contactInfo", "companyInfoLink"]
#    resultDf2 = pd.DataFrame(columns=columns)
    resultDf2 = pd.read_excel(r'E:\work_file\20190618_烟草局信息爬虫\result\induInfoSpide_weimao.xlsx')

    allLicense = data["licenseNum"].tolist()
    doneLicense = resultDf2["spideLicense"].tolist()
    preDoLicense = []
    for i in allLicense:
        if i not in doneLicense:
            preDoLicense.append(i)
    
    requests.packages.urllib3.disable_warnings()#解决requests因忽略证书验证而报错
    t = tqdm(range(len(preDoLicense)))
    faileTime = 0
    for i in t:
        retryTime = 0
        while retryTime<3:
            try:
#                licenseNum = data.loc[i, "licenseNum"]
                licenseNum = preDoLicense[i]
                req = visitWeb_req(licenseNum)
                df = parseHtml(req)
                resultDf2 = resultDf2.append(df)
                retryTime = 3
            except:
                retryTime +=1
                df = "nodata"
        if isinstance(df, str):
            faileTime += 1
        if faileTime >20:
            t.close()
            break
        if i%1000 ==0:
            resultDf2.to_excel(r'E:\work_file\20190618_烟草局信息爬虫\result\induInfoSpide_weimao_append.xlsx', index=False) 
    t.close()
    resultDf2.to_excel(r'E:\work_file\20190618_烟草局信息爬虫\result\induInfoSpide_weimao_append.xlsx', index=False)   


# =============================================================================
# selenium download font file
# =============================================================================
def visitWeb_sln(url="/company/detail/2580c15d5f7ce3a3f92dd5255fb806b1", driver=None):
    if not driver:
        driver = openBrowser()
    targetUrl = "https://www.weimao.com"+url
    driver.get(targetUrl)


def getFontLink(driver):
    html = driver.page_source
    pattern = re.compile("/\* IE6-IE8 \*/.*format\('woff2'\)", flags=re.DOTALL)
    pattern_sub = re.compile(".*url\('", flags=re.DOTALL)
    fontText = re.search(pattern, html).group().replace("') format('woff2')", "")
    fontText = re.sub(pattern_sub, "", fontText)
    return fontText


def downloadFont(driver):
    fontLink = getFontLink(driver)
    js='window.open("%s");' % fontLink
    driver.execute_script(js) ## 通过js打开一个新的窗口
    driver.get(fontLink)    
    
    
def transFontFile():
    filePath = r"C:/Users/hongzk/Downloads/下载"
    font = TTFont(filePath)    # 打开文件
    font.saveXML(r'E:/work_file/20190618_烟草局信息爬虫/result/fontFile/test.xml')     # 转换成 xml 文件并保存
    os.remove(filePath) #删除下载的字体文件




# =============================================================================
def strQ2B(ustring):
    pattern = re.compile(u'[，。：“”【】《》？；、（）‘’『』「」﹃﹄〔〕—·]')
    """中文特殊符号转英文特殊符号"""
    # 中文特殊符号批量识别
    fps = re.findall(pattern, ustring)
    # 对有中文特殊符号的文本进行符号替换

    if len(fps) > 0:
        ustring = ustring.replace(u'，', u',')
        ustring = ustring.replace(u'。', u'.')
        ustring = ustring.replace(u'：', u':')
        ustring = ustring.replace(u'“', u'"')
        ustring = ustring.replace(u'”', u'"')
        ustring = ustring.replace(u'【', u'[')
        ustring = ustring.replace(u'】', u']')
        ustring = ustring.replace(u'《', u'<')
        ustring = ustring.replace(u'》', u'>')
        ustring = ustring.replace(u'？', u'?')
        ustring = ustring.replace(u'；', u':')
        ustring = ustring.replace(u'、', u',')
        ustring = ustring.replace(u'（', u'(')
        ustring = ustring.replace(u'）', u')')
        ustring = ustring.replace(u'‘', u"'")
        ustring = ustring.replace(u'’', u"'")
        ustring = ustring.replace(u'’', u"'")
        ustring = ustring.replace(u'『', u"[")
        ustring = ustring.replace(u'』', u"]")
        ustring = ustring.replace(u'「', u"[")
        ustring = ustring.replace(u'」', u"]")
        ustring = ustring.replace(u'﹃', u"[")
        ustring = ustring.replace(u'﹄', u"]")
        ustring = ustring.replace(u'〔', u"{")
        ustring = ustring.replace(u'〕', u"}")
        ustring = ustring.replace(u'—', u"-")
        ustring = ustring.replace(u'·', u".")

    """全角转半角"""
    # 转换说明：
    # 全角字符unicode编码从65281~65374 （十六进制 0xFF01 ~ 0xFF5E）
    # 半角字符unicode编码从33~126 （十六进制 0x21~ 0x7E）
    # 空格比较特殊，全角为 12288（0x3000），半角为 32（0x20）
    # 除空格外，全角/半角按unicode编码排序在顺序上是对应的（半角 + 0x7e= 全角）,所以可以直接通过用+-法来处理非空格数据，对空格单独处理。

    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)  # 返回字符对应的ASCII数值，或者Unicode数值
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        try:
            lstr = chr(inside_code)
        except:
            lstr = chr(32)
        rstring += lstr  # 用一个范围在0～255的整数作参数，返回一个对应的字符
    return rstring


def collectInduChangeInfo():
    filePath = r"E:\work_file\20190618_烟草局信息爬虫\result\工商变更信息"
    fileList = [i for i in os.listdir(filePath) if i.endswith(".xlsx")]
    columns = ['company', 'date', 'changeItem', 'before', 'after']
    collectDf = pd.DataFrame(columns=columns)
    t = tqdm(range(len(fileList)))
    for i in t:
        file = fileList[i]
        companyName = file.replace(".xlsx", "")
        df = pd.read_excel(filePath+"\\"+file)
        df["company"] = companyName
        df = df[columns]
        collectDf = collectDf.append(df) #如果collectDf的列标签顺序和df列标签顺序不同，append后的新对象的列标签会打乱
    t.close()
    for col in collectDf.columns:
            collectDf[col] = collectDf[col].astype(str).apply(strQ2B)
    collectDf['date'] = collectDf['date'].str.replace("年", "-").str.replace("月", "-").str.replace("日", "")
    collectDf.to_excel(r'E:/work_file/20190618_烟草局信息爬虫/parseResult/工商变更信息_wm.xlsx', index=False)
            

# parse detail info
class parseHtml():
    def __init__(self, obj):
        self.obj = obj
        if isinstance(self.obj, selenium.webdriver.chrome.webdriver.WebDriver):
            self.soup = BeautifulSoup(self.obj.page_source , 'html.parser')
        elif isinstance(self.obj, requests.models.Response):
            self.soup = BeautifulSoup(self.obj.text, 'html.parser')
        self.body = self.soup.body
        self.companyName = self.body.find("div", attrs={"class":"summary-body"}).find("span").text
        self.columns = ["socialCreditCode", "taxpayerCode", "registerCode", "organizationCode", "companyType",
                "manageStatus", "legalPerson", "registerTime", "registerCapital", "manageTimeLimit",
                "registerOffice", "grantLicenseTime", "companyAddress", "businessRange",
                "keyPerson", "comment"]



    def getInfoNum(self):
        content = self.body.find("div", attrs={"class":"detail-nav no-wall"})
        info = []
        for record in content.find_all("a")[1:]: #第一个是基本信息
            item = record.text.replace("）", "").split("（")
            info.append(item[0]+"/"+item[1])
        return "+".join(info)



    def getInduInfo(self):
        content = self.body.find("div", attrs={"class":"detail-contain base-info"})
        
        socialCreditCode = content.find_all("tr")[0].find_all('td')[1].text
        taxpayerCode = content.find_all("tr")[0].find_all('td')[3].text
        registerCode = content.find_all("tr")[1].find_all('td')[1].text
        organizationCode = content.find_all("tr")[1].find_all('td')[3].text
        companyType = content.find_all("tr")[2].find_all('td')[1].text
        manageStatus = content.find_all("tr")[2].find_all('td')[3].text
        legalPerson = content.find_all("tr")[3].find_all('td')[1].text
        registerTime = content.find_all("tr")[3].find_all('td')[3].text
        registerCapital = content.find_all("tr")[4].find_all('td')[1].text
        manageTimeLimit = content.find_all("tr")[4].find_all('td')[3].text    
        registerOffice = content.find_all("tr")[5].find_all('td')[1].text
        grantLicenseTime = content.find_all("tr")[5].find_all('td')[3].text    
        companyAddress = content.find_all("tr")[6].find_all('td')[1].text  
        businessRange = content.find_all("tr")[7].find_all('td')[1].text 
        
        info = [socialCreditCode, taxpayerCode, registerCode, organizationCode, companyType,
                manageStatus, legalPerson, registerTime, registerCapital, manageTimeLimit,
                registerOffice, grantLicenseTime, companyAddress, businessRange]
        info = [re.sub("\s+", "", i) for i in info]
        return info
    


    def getKeyPerson(self):
        content = self.body.find("div", attrs={"id":"member"})
        if content:
            try:
                info = []
                for con in content.find_all("div", attrs={"class":"main-man-li"}):
                    name = con.find("a").text
                    position = con.find("p", attrs={"class":"position"}).text
                    types = con.find("span").text
                    record = name+"/"+position+"/"+types
                    info.append(record)
                info = "+".join(info)
                return info
            except:
                return "spider info wrong"
        else:
            return ""
        
    
    
    def getInduChange(self):
        content = self.body.find("div", attrs={"id":"change"})
        table = content.find("tbody", attrs={"id":"change_body"})
        info = []
        for record in table.find_all("tr"):
            date = record.find_all("td")[0].text
            changeItem = record.find_all("td")[1].text
            before = record.find_all("td")[2].text
            after = record.find_all("td")[3].text
            info.append([date, changeItem, before, after])
        df = pd.DataFrame(data=info, columns=['date', 'changeItem', 'before', 'after'])
        return df
    
    
    def getInvestment(self):
        content = self.body.find("div", attrs={"id":"invest_body"})
        companyList = content.find_all("div", attrs={"class":"company-li"})
        info = []
        for company in companyList:
            status = company.find("div", attrs={"class":"status"}).text
            companyName = company.find("div", attrs={"class":"search-left-content"}).find("a").text  
            companyInfoLink = company.find("div", attrs={"class":"search-left-content"}).find("a").get("href")
            basicInfo = company.find_all("span", attrs={"class":"info-li"})
            legalPerson = basicInfo[0].find_all("span")[1].text
            foundTime = basicInfo[1].find_all("span")[1].text #全角转半角
            registerCapital = basicInfo[2].find_all("span")[1].text
            companyAdress = company.find("div", attrs={"class":"company-adress"}).find("span").text
            record = [companyName, legalPerson, foundTime, registerCapital, companyInfoLink, status, companyAdress]
            info.append([re.sub("\s+", "", i) for i in record])
        df = pd.DataFrame(data=info, 
                          columns=['companyName', 'legalPerson', 'foundTime',
                                   'registerCapital', 'companyInfoLink', 'status', 'companyAdress'])
        return df



    def getBranchBody(self):
        content = self.body.find("ul", attrs={"id":"branch_body"})
        info = []
        for record in content.find_all("li"):
            company = record.find("div", attrs={"class":"margin-left-10"})
            text = [i for i in company.text.split("\n") if len(i)>0]
            companyName = text[0]
            legalPerson = '+'.join(text[1:])
            info.append([companyName, legalPerson])
        df = pd.DataFrame(data=info, columns=['branchCompany', 'legalPerson'])
        return df

    
    
    def getHtmlInfo(self):
        induInfo = self.getInduInfo() #工商信息
#        index = induInfo[0] + self.companyName
        itemInfoNum = {}
        for i in self.body.find("div", attrs={"class":"ui tab anchor-nav active"}).find_all("a")[2:6]:
            item = i.text.split(" ")
            item = [j for j in item if len(j)>0]
            itemInfoNum[item[0]] = int(item[1])
        if itemInfoNum["主要人员"] > 0:
            try:
                keyPerson = self.getKeyPerson()
            except:
                keyPerson = "spider failed(have data)"
        else:
            keyPerson = ""
            
        comment = ""
        if itemInfoNum["工商变更"] > 0:
            try:
                induChange = self.getInduChange()
                induChange["socialCreditCode"] = induInfo[0]
                induChange["companyName"] = self.companyName
            except:
                comment += "工商变更 spider failed(have data)"
                induChange = pd.DataFrame()
        else:
            induChange = pd.DataFrame()
        if itemInfoNum["对外投资"] > 0:
            try:
                investment = self.getInvestment()
                investment["socialCreditCode"] = induInfo[0]
                investment["companyName"] = self.companyName
            except:
                comment += "对外投资 spider failed(have data)"
                investment = pd.DataFrame()
        else:
            investment = pd.DataFrame()
        if itemInfoNum["分支机构"] > 0:
            try:
                branchBody = self.getBranchBody()
                branchBody["socialCreditCode"] = induInfo[0]
                branchBody["companyName"] = self.companyName
            except:
                branchBody = pd.DataFrame()
                comment += "分支机构 spider failed(have data)"
        else:
            branchBody = pd.DataFrame()
        result = induInfo+[keyPerson, comment]
        df = pd.DataFrame(data=[result], columns=self.columns)
        return df, induChange, investment, branchBody


def openBrowser(ip=False, headless=False):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation']) #伪装window.navigator.webdriver为undefined
    if not ip:
        pass
    else:
        chrome_options.add_argument("--proxy-server=http://%s"%ip) #添加代理IP
    if headless:
        chrome_options.add_argument("headless")
    else:
        pass
#    option.add_argument('--user-data-dir=C:/Users/hongzk/AppData/Local/Google/Chrome/User Data') #注意斜杠和反斜杠
#    chrome_options.add_argument('--disable-extensions')
#    chrome_options.add_argument('--profile-directory=Default')
#    chrome_options.add_argument("--incognito")
#    chrome_options.add_argument("--disable-plugins-discovery");
#    chrome_options.add_argument("--start-maximized")
#    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")#需提前关闭所有chrome
    driver = webdriver.Chrome(options=chrome_options) 
    return driver

   

if __name__ == "__main__":
    saveFilePath = r'E:/work_file/20190712_微猫爬虫/result/'
    
    linkData = pd.read_excel(r'E:/work_file/20190618_烟草局信息爬虫/result/induInfoSpide_weimao.xlsx')
    linkData_unique = linkData.drop_duplicates(subset="companyInfoLink")
    df_induChange = pd.DataFrame()
    df_investment = pd.DataFrame()
    df_branchBody = pd.DataFrame()
    collectData = pd.DataFrame()
    error = []

#    availableIP = """114.229.11.38:4207
#183.166.125.59:4231
#183.166.7.131:4251
#115.219.73.253:4217
#113.117.64.220:4228
#180.122.88.93:4236
#60.189.159.172:4257
#124.152.85.120:4264
#122.233.27.60:4263
#114.239.89.197:4276""".split("\n")
#    randomIndex = random.randint(0, len(availableIP) - 1)
#    ip = availableIP[randomIndex]
    ip = "182.34.197.161:4246"
    driver = openBrowser(ip=ip)
    driver.get("https://www.weimao.com/")
    #添加一个自动登陆的脚本
    
    cookies = driver.get_cookies()
    print(cookie_list)
    t = tqdm(range(20))
    for i in t:
        data = linkData.loc[i, :].tolist()
        try:
            visitWeb_sln(url=data[0], driver=driver)
            test = parseHtml(driver)
            df, induChange, investment, branchBody = test.getHtmlInfo()
            collectData = collectData.append(df)
            df_induChange = df_induChange.append(induChange)
            df_investment = df_investment.append(investment)
            df_branchBody = df_branchBody.append(branchBody)
    
        except:
            print("spider has someting wrong")
            print(data[-1])
            print(data[0])
            error.append(data[-1])
        if len(error)>10:
            break
        if i%50==0:
            collectData.to_excel(saveFilePath+'weimao_detailInfo.xlsx', index=False)
            df_induChange.to_excel(saveFilePath+'weimao_induChange.xlsx', index=False)
            df_investment.to_excel(saveFilePath+'weimao_investment.xlsx', index=False)
            df_branchBody.to_excel(saveFilePath+'weimao_branchBody.xlsx', index=False)

    cookie_list.append(cookies)
    print(cookie_list)
    #爬虫数据处理
    collectInduChangeInfo()

#    t.close()
