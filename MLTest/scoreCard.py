# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:29:21 2019

@author: hongzk
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  # 切分训练集测试集
from sklearn import metrics
from xgboost import XGBClassifier as XGBC

# 中文乱码和坐标轴负号的处理
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']#
plt.rcParams['font.sans-serif']=['SimHei'] #用于显示图片中文
plt.rcParams['axes.unicode_minus'] = False


os.chdir(r'/Users/hongzk/Documents/data_files')
orgData = pd.read_hdf('allData.hdf')
#orgData = orgData.sample(30000)
data = orgData.sample(30000)

data.info()
colTypes = pd.DataFrame(data.dtypes).reset_index()
colTypes.columns = ['columns', 'dtype']
colTypes.groupby(['dtype']).count()

var_int = colTypes[colTypes['dtype'] == 'int64']['columns'].tolist()
var_float = colTypes[colTypes['dtype'] == 'float64']['columns'].tolist()
var_obj = colTypes[colTypes['dtype'] == 'object']['columns'].tolist()
data[var_obj].head()

data.rename(columns={"is_suspend_days_gt60": "target"}, inplace=True)
df_target = data[['cust_code', 'target']]
data.drop(["target", "cust_code"], axis=1, inplace=True)
data.fillna(0, inplace=True)

def drop_corr(df, threshold):
    """
    根据相关系数删除变量
    """
    df_corr = df.corr().abs()
    corr_index = np.where(df_corr >= threshold)
    drop_cols = [df_corr.columns[y] for x, y in zip(*corr_index)
                 if x != y and x < y]
    df_left = df.loc[:, ~df.columns.isin(drop_cols)]
    return df_left

df_drop_corr = drop_corr(data, 0.9)
cols_corr = df_drop_corr.columns.tolist()


#随机森林挑选变量
def select_by_rf(df, cols, y, threshold):
    x_train = df[cols]
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y)
    importances = rfc.feature_importances_
#    indices = np.argsort(importances)[::-1]
    x_selected = x_train.iloc[:, importances > threshold]
    return x_selected

df_drop_rf = select_by_rf(df_drop_corr, [i for i in cols_corr if i not in var_obj], df_target['target'], 0.001)
cols_rf = df_drop_rf.columns.tolist()


def cal_iv(data, y, bin_cols):
    #计算woe、iv
    bad_weight, good_weight = 1, 1
    qushi = "up"
    df_iv = pd.DataFrame()
    for i in bin_cols:
        trans_df = pd.crosstab(data[i], y)
        trans_df = trans_df.rename(
            columns={0: 'good_count', 1: 'bad_count', '0': 'good_count', '1': 'bad_count'})
        trans_df["total"] = trans_df["good_count"] + trans_df["bad_count"]
        trans_df['good_count_weight'] = trans_df['good_count'] * good_weight  # 每条bin特征字段对应的好客户数量*好客户权重(默认1)
        trans_df['bad_count_weight'] = trans_df['bad_count'] * bad_weight  # 每条bin特征字段对应的坏客户数量*坏客户权重(默认1)
        trans_df['total_weight'] = trans_df['good_count_weight'] + trans_df['bad_count_weight']  # 权重相加
    
        good_total = trans_df['good_count_weight'].sum()  # 好客户权重汇总
        bad_total = trans_df['bad_count_weight'].sum()  # 坏客户权重汇总
        all_ = good_total + bad_total  # 权重和
        trans_df['bin_pct'] = trans_df['total_weight'] / all_  # 所占比例
    
        trans_df["bad_rate"] = trans_df['bad_count_weight'].div(trans_df['total_weight'])
        trans_df['sample_bad_rate'] = bad_total / all_  # 所有坏客户的占比
        good_dist = np.nan_to_num(trans_df['good_count_weight'] / good_total)  # nan值用0替代inf值用有限值替代
        bad_dist = np.nan_to_num(trans_df['bad_count_weight'] / bad_total)
    
        trans_df['woe'] = np.log(bad_dist / good_dist)
        trans_df['woe'] = round(trans_df['woe'], 4)
        trans_df['iv_i'] = (bad_dist - good_dist) * trans_df['woe']
        col_iv = trans_df['iv_i'].sum()
        tmp = pd.DataFrame(data=[[i, col_iv]], columns=["colName", "iv"])
        df_iv = df_iv.append(tmp)
    return df_iv


def select_by_iv(df, colNums):
    bin_cols = [i for i in df.columns if i.endswith("_bin")]
    df_iv = cal_iv(df, df_target["target"], bin_cols)
    col_drop_iv = df_iv.sort_values(by="iv", ascending=False).head(colNums)["colName"].tolist()
    col_drop_iv = [i[:-4] for i in col_drop_iv]
    return df[col_drop_iv]

#df_drop_rf.drop("cust_code", axis=1, inplace=True)
for col in cols_rf:
    df_drop_rf[col+"_bin"] = pd.qcut(df_drop_rf[col], 5, duplicates='drop')
df_drop_iv = select_by_iv(df_drop_rf, 8)
cols_iv = df_drop_iv.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(data[cols_iv], df_target['target'], test_size=0.3)

xgb = XGBC()
xgb.fit(X_train, y_train)
y_predict = xgb.predict(X_test)

def map_bin(x, bins):
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True
    the_bin = bins[np.digitize([x], bins, **kwargs)[0]]
    bin_lower = bins[np.digitize([x], bins, **kwargs)[0] - 1]
    return '{0}~{1}'.format(round(bin_lower, 4), round(the_bin, 4))


def decision_split(var, y):
    min_samples_leaf = min(len(y) / 10, 100)
    dt = DecisionTreeClassifier(criterion='entropy',
                                max_depth=4,
                                max_leaf_nodes=8,
                                min_samples_leaf=min_samples_leaf)
    var2 = var.values.reshape(-1, 1)
    dt.fit(var2, y)
    bins = np.unique(dt.tree_.threshold[dt.tree_.feature > -2])
    if len(bins) > 0:
        bins[0] = -np.inf
        bins[-1] = np.inf
        bucket = var.apply(map_bin, bins=bins)
    else:
        bucket = None
    return bucket, bins


binsDict = {}
for col in cols_iv:
    X_train[col + '_bin'], bins = decision_split(X_train[col], y_train)
    binsDict[col] = bins
    
model_vars = [i for i in X_train.columns if i.endswith("_bin")]
X_train = X_train[model_vars]
for i in model_vars:
    bin_nums = len(X_train[i].unique())
    if bin_nums<=3:
        X_train.drop(i, axis=1, inplace=True)


def map_woe(data, y, bin_cols):
    #计算woe、iv
    bad_weight, good_weight = 1, 1
    qushi = "up"
    colNames = []
    for i in bin_cols:
        trans_df = pd.crosstab(data[i], y)
        trans_df = trans_df.rename(
            columns={0: 'good_count', 1: 'bad_count', '0': 'good_count', '1': 'bad_count'})
        trans_df["total"] = trans_df["good_count"] + trans_df["bad_count"]
        trans_df['good_count_weight'] = trans_df['good_count'] * good_weight  # 每条bin特征字段对应的好客户数量*好客户权重(默认1)
        trans_df['bad_count_weight'] = trans_df['bad_count'] * bad_weight  # 每条bin特征字段对应的坏客户数量*坏客户权重(默认1)
        trans_df['total_weight'] = trans_df['good_count_weight'] + trans_df['bad_count_weight']  # 权重相加
    
        good_total = trans_df['good_count_weight'].sum()  # 好客户权重汇总
        bad_total = trans_df['bad_count_weight'].sum()  # 坏客户权重汇总
        all_ = good_total + bad_total  # 权重和
        trans_df['bin_pct'] = trans_df['total_weight'] / all_  # 所占比例
    
        trans_df["bad_rate"] = trans_df['bad_count_weight'].div(trans_df['total_weight'])
        trans_df['sample_bad_rate'] = bad_total / all_  # 所有坏客户的占比
        good_dist = np.nan_to_num(trans_df['good_count_weight'] / good_total)  # nan值用0替代inf值用有限值替代
        bad_dist = np.nan_to_num(trans_df['bad_count_weight'] / bad_total)
    
        trans_df['woe'] = np.log(bad_dist / good_dist)
        trans_df['woe'] = round(trans_df['woe'], 4)
        trans_df = trans_df.reset_index()
        mapDict = dict(zip(trans_df[i], trans_df['woe']))
        colName = i.replace("_bin", "_woe")
        colNames.append(colName)
        data[colName] = data[i].map(mapDict) 
    return data[colNames]

X_train = map_woe(X_train, y_train, X_train.columns)

lr = LogisticRegression()
lr.fit(X_train, y_train)


def modEffect(y_test, y_predict):
    acc = metrics.accuracy_score(y_test, y_predict)
    precision = metrics.precision_score(y_test, y_predict)
    recall = metrics.recall_score(y_test, y_predict)
    f1 = metrics.f1_score(y_test, y_predict)
    auc= metrics.roc_auc_score(y_test, y_predict)
    print('准确率:{:.4f},精确率:{:.4f},召回率:{:.4f},f1-score:{:.4f},auc:{:.4f}'.format(acc, precision, recall, f1, auc))

modEffect(y_test, y_predict)
fpr,tpr,threshold = metrics.roc_curve(y_test, y_predict)
# 计算AUC的值
roc_auc = metrics.auc(fpr,tpr)
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')  # 绘制面积图
plt.plot(fpr, tpr, color='black', lw = 1)  # 添加边际线
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')  # 添加对角线
plt.text(0.5,0.3,'ROC curve (area = %0.3f)' % roc_auc)  # 添加文本信息
plt.xlabel('1-Specificity')  # 添加x轴与y轴标签
plt.ylabel('Sensitivity')
plt.show()


# 网格搜索
from sklearn.model_selection import GridSearchCV
parameters = {'gamma': [0.001, 0.01, 0.1, 1], 'C':[0.001, 0.01, 0.1, 1,10]}
gs = GridSearchCV(svc, parameters, refit = True, cv = 5, verbose = 1, n_jobs = -1)
gs.fit(X_train, y_train)
print('最优参数: ',gs.best_params_)
print('最佳性能: ', gs.best_score_)
y_predict = gs.predict(X_test)
modEffect(y_test, y_predict)