# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:06:01 2019

@author: hongzk
"""

#utils
import pandas as pd
import numpy as np
import sqlalchemy
from selenium import webdriver

dbConnection = {
        'risk':["10.0.18.34:3306/risk_dept_report", "antian", "Antian@2018"],
        'dev_ex':["10.0.18.34:3306/dev_iccs_ex", "antian", "Antian@2018"],
        'dev_cu':["10.0.18.34:3306/dev_iccs_cu", "antian", "Antian@2018"],
        'dev_ca':["10.0.18.34:3306/dev_iccs_ca", "antian", "Antian@2018"]
        }


def mysqlEngine(database):
    connectInfo = dbConnection[database]
    IP = connectInfo[0].split("/")[0]
    dbName = connectInfo[0].split("/")[1]
    user = connectInfo[1]
    password = connectInfo[2]
    # mysql+pymysql://<user>:<password>@<host>/<dbname>?charset=utf8
    string = "mysql+pymysql://%s:%s@%s/%s?charset=utf8" % (user, password, IP, dbName)
    engine = sqlalchemy.create_engine(string)
    return engine


def set_dtype(df, sp=[]):
    column_types = {}
    dtypes = df.dtypes
    for i, k in enumerate(dtypes.index):
        dt = dtypes[k]
        print(dt)
        if k in sp:
            size = 2 + df[k].astype(str).map(len).max()
            column_types[k] = sqlalchemy.String(length=size)
        elif str(dt.type) == "<type 'numpy.datetime64'>":
            column_types[k] = sqlalchemy.DateTime
        elif issubclass(dt.type, np.datetime64):
            column_types[k] = sqlalchemy.DateTime
        elif issubclass(dt.type, (np.integer, np.bool_)):
            column_types[k] = sqlalchemy.Integer
        elif issubclass(dt.type, np.floating):
            column_types[k] = sqlalchemy.Float
        elif issubclass(dt.type, np.object_):
            size = 2 + df[k].astype(str).map(len).max()
            column_types[k] = sqlalchemy.String(length=size)
        else:
            sampl = df[df.columns[i]][0]
            if str(type(sampl)) == "<type 'datetime.datetime'>":
                column_types[k] = sqlalchemy.DateTime
            elif str(type(sampl)) == "<type 'datetime.date'>":
                column_types[k] = sqlalchemy.DateTime
            else:
                size = 2 + df[k].astype(str).map(len).max()
                column_types[k] = sqlalchemy.String(length=size)
    return column_types


class MySQLTool(object):
    def __init__(self, dbName):
        self.dbName = dbName
        self.con = mysqlEngine(dbName)


    def read(self, sql):
        try:
            df = pd.read_sql(con=self.con, sql=sql)
        except:
            print('请输入正确信息')
            df = None
        return df



    def insert(self, df=None, table=None, sp=[], if_exists='append'):
        dtypes = set_dtype(df, sp=sp)
        try:
            df.to_sql(table, con=self.con, if_exists=if_exists, dtype=dtypes, index=False, chunksize=100)
        except:
            print('该库不支持插入')



    def key_insert(self, df=None, table=None, sp=[], if_exists='replace', primary_keys=None):
        dtypes = set_dtype(df, sp=sp)
        try:
            df.to_sql(table, con=self.con, if_exists=if_exists, dtype=dtypes, index=False, chunksize=100)
            insp = sqlalchemy.inspect(self.dbName)
            if not insp.get_primary_keys(table):
                with self.con.connect() as con:
                    con.execute('ALTER  TABLE `%s` ADD PRIMARY KEY (%s);' % (table, primary_keys))
        except Exception:
            print('正在插入重复数据，默认跳过')
            pass
        


def openBrowser():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation']) #伪装window.navigator.webdriver为undefined
#    chrome_options.add_argument("--proxy-server=http://%s"%ip) #添加代理IP
#    option.add_argument('--user-data-dir=C:/Users/hongzk/AppData/Local/Google/Chrome/User Data') #注意斜杠和反斜杠
#    chrome_options.add_argument('--disable-extensions')
#    chrome_options.add_argument('--profile-directory=Default')
#    chrome_options.add_argument("--incognito")
#    chrome_options.add_argument("--disable-plugins-discovery");
#    chrome_options.add_argument("--start-maximized")
#    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")#需提前关闭所有chrome
    driver = webdriver.Chrome(options=chrome_options) 
    return driver


def visitWeb_sln(url="http://139.159.237.175:8888/accounts/login/?next=/hue/editor/%3Ftype%3Dimpala", driver=None):
    if not driver:
        driver = openBrowser()
    targetUrl = url
    driver.get(targetUrl)


def SignIn(driver):
    id = "hongzk"
    password = "88888888"
    input_id = driver.find_element_by_id("id_username")
    input_id.clear()
    input_id.send_keys(id)
    input_password = driver.find_element_by_id("id_password")
    input_password.clear()
    input_password.send_keys(password)
    driver.find_element_by_class_name("btn btn-primary")#wrong


driver.find_element_by_tag_name()
driver = openBrowser()
visitWeb_sln(driver=driver)
input_selectCode = driver.find_element_by_class_name("ace_layer ace_marker-layer")
input_selectCode.send_keys("select * from model.variables_set_drawback limit 1000")
