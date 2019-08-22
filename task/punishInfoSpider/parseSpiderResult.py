# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:57:08 2019

@author: hongzk
"""

import pandas as pd
import numpy as np
import re

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

def parseName(name, whichName=1):
    splitName = name.split("(")
    nameDict = {}
    if len(splitName)==1:
        if len(splitName[0])<=3:
            nameDict["管理者"] = splitName[0]
        else:
            nameDict["店名"] = splitName[0]
    else:
        nameDict["管理者"] = splitName[0]
        nameDict["店名"] = splitName[1]
    
    if whichName==1:
        return nameDict.get("管理者", None)
    elif whichName==2:
        return nameDict.get("店名", None)
    else:
        print("input wrong whichName!")


orgData = pd.read_excel(r'E:/work_file/20190618_烟草局信息爬虫/result/custPunishInfo_sh.xlsx')
data = orgData[['立案时间', '当事人', '案件名称', '业务时间', '处罚决定', '社会信用代码', '处罚日期']]

for col in ["当事人", "案件名称", "处罚决定"]:
    data[col] = data[col].map(strQ2B)

data["当事人"] = data["当事人"].apply(lambda x: re.sub("\)$", "", x))
data["管理者"] = data["当事人"].apply(lambda x:parseName(x, whichName=1))
data["店名"] = data["当事人"].apply(lambda x:parseName(x, whichName=2))

data = data[['社会信用代码', '管理者', '店名', '立案时间', '业务时间', '处罚日期', "案件名称", '处罚决定']]
data.to_excel(r'E:/work_file/20190618_烟草局信息爬虫/parseResult/custPunishInfo_sh.xlsx', index=False)



# =============================================================================
# 
# =============================================================================
import pickle

fr = open(r'E:/work_file/20190618_烟草局信息爬虫/result/违法信息_湖南.pkl', 'rb')
resultDict = pickle.load(fr)






