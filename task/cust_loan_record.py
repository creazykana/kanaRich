# -*- coding: utf-8 -*-
"""
Created on Sun May  5 10:03:34 2019

@author: hongzk
"""

#客户借款记录表
import os
os.chdir(r'E:\GIT\AntiyWork')
import pandas as pd
import numpy as ny
import sqlalchemy
from datetime import datetime
from utils import dbConnection,MySQLTool

ms = MySQLTool('dev_ca')
sql = 'select * from dev_iccs_ca.loan_info limit 1000'
df = ms.read(sql)
for i in range(len(df)):
    begin = datetime.strptime(df.loc[i, 'loanbegin'],"%Y%m%d")
    end = datetime.strptime(df.loc[i, 'loanend'], "%Y%m%d")
    timeDiff = begin-end







    begin = datetime.strptime("20180228","%Y%m%d")
    end = datetime.strptime("20190608", "%Y%m%d")
    begin_year = begin.year
    begin_month = begin.month
    end_year = end.year
    end_month = end.month
    
    yearDiff = end_year-begin_year
    monthDiff = end_month-begin_month
    timeDiff = yearDiff*12+monthDiff-1
    
    new_time = datetime.strptime(str(begin_year+yearDiff)+str(begin_month+monthDiff-1)+str(begin.day), "%Y%m%d") #可能会出错
    dayDiff = (end-new_time).days
    timeDiff.days
