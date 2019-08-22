from bs4 import BeautifulSoup
import requests
import pandas as pd
from tqdm import tqdm
import time

requests.packages.urllib3.disable_warnings()#解决requests因忽略证书验证而报错

# =============================================================================
# 反爬机制：登陆；300左右访问出现图片验证码
# =============================================================================





def linkWeb(licenseNum, cookies):
    targetUrl = "https://xin.baidu.com/s"
    searchDict = {"q":licenseNum, "t":"0"}
    req = requests.get(targetUrl, cookies=cookies, params=searchDict, verify=False, timeout=1)
    if req.status_code==200:
        pass
    else:
        print("Waring: %s requests status code is error(value is %s)"% (licenseNum, req.status_code))
    return req


def getSimpleInfo(req):
    soup = BeautifulSoup(req.text, 'html.parser')
    body = soup.body
    resultNum = body.find("div", attrs={"class":"zx-list-count-left"}).find("em").text#body.select("em[class^=zx-result-counter]")
    if int(resultNum)<1:
        return ["", "", "", "", "", "", "", ""]
    else:
        return parseHtml(body)
    
    
def parseHtml(body):
    result = body.find("div", attrs={"class":"zx-ent-items"})
    companyName = result.find("h3").find('a').text
    detailInfoLink = result.find("h3").find('a').get("href") # https://xin.baidu.com/detail
    registerCode = result.find("span", attrs={"class":"zx-ent-hit-reason-text"}).find("em").text
    legalPerson = result.find("span", attrs={"class":"legal-txt"}).text
    registerCapital = result.find_all("span", attrs={"class":"zx-ent-item zx-ent-text middle"})[1].text
    foundTime = result.find("span", attrs={"class":"zx-ent-item zx-ent-text short"}).text
    address = result.find("span", attrs={"class":"zx-ent-item zx-ent-text long"}).text
    busiScope = result.find("span", attrs={"class":"zx-ent-item zx-ent-text expand"}).text
    info = [companyName, detailInfoLink, registerCode, legalPerson, registerCapital, foundTime, address, busiScope]
    return info



def run():
    orgData = pd.read_excel(r"E:/work_file/20190618_烟草局信息爬虫/data/深圳执照号名单.xlsx")
    data = orgData[["cust_code", "business_license_number_dc", "register_area"]]
     
    colNames = ["cust_code", "licenseNum", "registerArea", "companyName", "detailInfoLink", 
                 "registerCode", "legalPerson", "registerCapital", "foundTime", "address", "busiScope"]
    resultDf = pd.DataFrame(columns=colNames)
    
    cookies = {"BAIDUID":"5A1215A27D1529797FC622FF67060E0A:FG=1", 
                "BIDUPSID":"5A1215A27D1529797FC622FF67060E0A", 
                "PSTM":"1544601747", 
                "pgv_pvi":"5094735872", 
                "BDPPN":"7e4d61a72aae8bcdd72dc18968fb3fd8", 
                "log_guid":"284d072a5fa6d015460217918b1e7ee0", 
                "Hm_lvt_baca6fe3dceaf818f5f835b0ae97e4cc":"1561690665,1561948837", 
                "__cas__st__":"NLI",
                "__cas__id__":"0", 
                "ZX_UNIQ_UID":"13cf0694c94d8223d1357562f22ed184", 
                "ZX_HISTORY":"%5B%7B%22visittime%22%3A%222019-07-01+15%3A56%3A55%22%2C%22pid%22%3A%22xlTM-TogKuTwP5k-TcnxUlw9g5JbypU4jQmd%22%7D%2C%7B%22visittime%22%3A%222019-07-01+10%3A41%3A26%22%2C%22pid%22%3A%22xlTM-TogKuTwfG2w1sKdLtE3QRs3wZ1SCAmd%22%7D%5D", 
                "delPer":"0", 
                "H_PS_PSSID":"1466_21107_18560_29135_29238_28519_29099_28834_29220_26350_28704", 
                "BDUSS":"nhPVlUtdFZMZlJnMkFObVFnN1lSRmEwaHNmLU5rVDJZUHRKLXdHd1NjRk1Wa0ZkSVFBQUFBJCQAAAAAAAAAAAEAAADNK0qMR2hvdWxfSEsAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEzJGV1MyRldd", 
                "Hm_lpvt_baca6fe3dceaf818f5f835b0ae97e4cc":"1561971025"}
    t = tqdm(range(500))
    for i in t:
        licenseNum = data.loc[i, "business_license_number_dc"]
        s = linkWeb(licenseNum, cookies)
        try:
            licenseInfo = getSimpleInfo(s)
        except:
            licenseInfo = ["wrong", "wrong", "wrong", "wrong", "wrong", "wrong", "wrong", "wrong", ]
        record = data.loc[i, :].tolist()+licenseInfo
        df = pd.DataFrame(data=[record], columns=colNames)
        resultDf = resultDf.append(df)
        time.sleep(0.1)
    t.close()
    resultDf = resultDf.reset_index(drop=True)

if __name__=="__main__":
        
        