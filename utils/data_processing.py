#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:48:13 2017

@author: shanjie
"""

import pandas as pd
import numpy as np
from itertools import groupby
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from sklearn.preprocessing import Imputer
import pickle

_province = ('北京市', '上海市', '香港', '台湾', '重庆市', '澳门', '天津市', '江苏省', '浙江省',
             '四川省', '江西省', '福建省', '青海省', '吉林省', '贵州省', '陕西省', '山西省',
             '河北省', '湖北省', '辽宁省', '湖南省', '山东省', '云南省', '河南省', '广东省',
             '安徽省', '甘肃省', '海南省', '黑龙江省', '内蒙古自治区', '新疆维吾尔自治区',
             '广西壮族自治区', '宁夏回族自治区', '西藏自治区')
_province_slice = ('北京', '上海', '香港', '台湾', '重庆', '澳门', '天津', '江苏', '浙江',
                   '四川', '江西', '福建', '青海', '吉林', '贵州', '陕西', '山西',
                   '河北', '湖北', '辽宁', '湖南', '山东', '云南', '河南', '广东',
                   '安徽', '甘肃', '海南', '黑龙江', '内蒙古', '新疆',
                   '广西', '宁夏', '西藏')
_contract_code = {'2010': '待审核',
                  '2020': '审核中',
                  '3000': '通过-已成单',
                  '3001': '拒绝',
                  '3002': '撤销',
                  '3010': '通过-签署',
                  '3020': '通过-激活',
                  '3030': '通过-正常结清',
                  '3031': '通过-提前结清',
                  '3032': '通过—逾期结清',
                  '3033': '通过-退货',
                  '4040': '中间状态，信审人员锁定'}
_jxl30000 = ("权限", "锁定", "被保护", "已锁", "系统", "为空", "认证", "异常", "网站", "运营商", "繁忙", "不允许", "失败")


def save(obj, name):
    """
    导出为pickle文件
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read(name):
    df = pd.read_pickle(name + '.pkl')
    return df


def set_mean(data, ftr):
    series = data[ftr]
    tmp = Imputer(axis=1, strategy='median').fit_transform(series)
    data[ftr].update(pd.Series(tmp[0]))


def set_median(data, ftr):
    series = data[ftr]
    tmp = Imputer(axis=1, strategy='median').fit_transform(series)
    data[ftr].update(pd.Series(tmp[0]))


def ployinterp_column(s, n, k=5):
    from scipy.interpolate import lagrange
    y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)


def interp_df(df):
    """
    拉格朗日插值
    """
    for i in df.columns:
        for j in range(len(df)):
            if (df[i].isnull())[j]:  # 如果是空值就插值
                df[i][j] = ployinterp_column(df[i], j)
    return df


def move_target_last(data, target_col):
    reindex_col = [c for c in data.columns]
    if target_col not in reindex_col:
        return data
    reindex_col.remove(target_col)
    reindex_col.append(target_col)
    return data.reindex_axis(reindex_col, axis=1)


def cal_cv(x):
    temp = x.dropna()
    count = len(temp)
    mean = temp.mean()
    if count == 0:
        cv = -999
    elif count == 1:
        cv = -998
    else:
        if mean == 0:
            cv = 999
        else:
            std = temp.std()
            cv = std / mean
    return cv


def continue_increasing(l):
    temp = (x < y for x, y in zip(l, l[1:]))
    grouped_temp = [sum(1 for i in g) for k, g in groupby(temp) if k]
    if len(grouped_temp) == 0:
        result = np.nan
    else:
        result = max(grouped_temp)
    return result


def continue_increasing_nearly(l):
    temp = (x < y for x, y in zip(l, l[1:]))
    grouped_temp = [sum(1 if k else -1 for i in g) for k, g in groupby(temp)]
    if len(grouped_temp) == 0:
        result = np.nan
    else:
        result = grouped_temp[-1]
    return result


def increasing_count(l):
    return sum(x < y for x, y in zip(l, l[1:]))


def decreasing_count(l):
    return sum(x > y for x, y in zip(l, l[1:]))


def count_na_nearly(l):
    temp = l.fillna('na')
    grouped_temp = [sum(1 if k == 'na' else 0 for i in g) for k, g in groupby(temp)]
    if len(grouped_temp) == 0:
        result = np.nan
    else:
        result = grouped_temp[-1]
    return result


def count_na_far(l):
    temp = l.fillna('na')
    grouped_temp = [sum(1 if k == 'na' else 0 for i in g) for k, g in groupby(temp)]
    if len(grouped_temp) == 0:
        result = np.nan
    else:
        result = grouped_temp[0]
    return result


def divall(a, b):
    if b:
        result = np.divide(a, b)
    elif np.isnan(a) or np.isnan(b):
        result = -999
    elif a == 0 and b == 0:
        result = -9999
    elif b == 0:
        result = 9999
    else:
        result = 'error'
    return result


def allsum(l):
    if len(l) == 0:
        result = np.nan
    else:
        result = np.nansum(l)
    return result


def match(first, second):
    if len(first) == 0 and len(second) == 0:
        return True
    if len(first) > 1 and first[0] == '*' and len(second) == 0:
        return False
    if (len(first) > 1 and first[0] == '?') or (len(first) != 0
                                                and len(second) != 0 and first[0] == second[0]):
        return match(first[1:], second[1:]);
    if len(first) != 0 and first[0] == '*':
        return match(first[1:], second) or match(first, second[1:])
    return False


def data_trans(data, target, del_columns=None):
    """
    data: train data
    target: label
    del_columns: drop columns
    """
    transformed = data.copy()
    if del_columns:
        del_cols = [i for i in del_columns if i in transformed.columns]
    else:
        del_cols = []
    transformed.drop(del_cols, axis=1, inplace=True)
    contineous_describe = transformed.describe()
    non_features = set([target]) | set(del_cols)
    continueous_features = set(contineous_describe.columns) - non_features
    categorical_features = set(transformed.columns) - set(continueous_features) - non_features
    for feature in categorical_features:
        transformed[feature] = transformed[feature].astype('category')
    for feature in continueous_features:
        transformed[feature] = transformed[feature].astype('float32')
    return transformed, categorical_features, continueous_features


def dropna_df(df, columns, pct=0.15):
    """
    根据缺失率删除变量
    """
    not_na_count = df[columns].count()
    pct_series = not_na_count / len(df)
    remain_series = pct_series[pct_series >= pct]
    remain_columns = remain_series.index.tolist()
    remain_df = df.loc[:, remain_columns]
    return remain_df


def count_row_null(df):
    """
    统计行缺失率
    """
    null_row = pd.isnull(df).sum(axis=1)
    count_row = df.shape[1]
    null_rate = null_row / count_row
    return null_rate


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


def des(df):
    """
    数据类型及缺失情况 by shan
    """
    dtypes = df.dtypes
    columns = ['dtype', 'count', 'missing', 'missing_rate', 'unique', 'top', 'freq', 'mean',
               'std', 'min', 'pct_5', 'pct_25', 'pct_50', 'pct_75', 'pct_95', 'max']
    des_df = pd.DataFrame(columns=columns)
    for k in dtypes.index:
        dt = str(dtypes[k])
        if 'datetime' in dt:
            temp = df.select_dtypes(include=['datetime']).describe().T \
                .assign(dtype='datetime',
                        missing=df.apply(lambda x: len(x) - x.count()),
                        missing_rate=df.apply(lambda x: (len(x) - x.count()) / len(x)))
        elif dt == 'object':
            temp = df.select_dtypes(include=['O']).describe().T \
                .assign(dtype='object',
                        missing=df.apply(lambda x: len(x) - x.count()),
                        missing_rate=df.apply(lambda x: (len(x) - x.count()) / len(x)))
        elif dt == 'float64' or dt == 'int64':
            temp = (df.select_dtypes(include=['number']).describe().T) \
                .drop(['25%', '50%', '75%'], axis=1) \
                .assign(dtype=dt,
                        missing=df.apply(lambda x: len(x) - x.count()),
                        missing_rate=df.apply(lambda x: (len(x) - x.count()) / len(x)),
                        unique=df.apply(lambda x: x.nunique()),
                        mean=df.mean(),
                        pct_5=df.select_dtypes(include=['number']).apply(lambda x: x.dropna().quantile(.05)),
                        pct_25=df.select_dtypes(include=['number']).apply(lambda x: x.dropna().quantile(.25)),
                        pct_50=df.select_dtypes(include=['number']).apply(lambda x: x.dropna().quantile(.5)),
                        pct_75=df.select_dtypes(include=['number']).apply(lambda x: x.dropna().quantile(.75)),
                        pct_95=df.select_dtypes(include=['number']).apply(lambda x: x.dropna().quantile(.95)))
        elif dt == 'bool':
            temp = df.select_dtypes(include=['bool']).describe(include=('all')).T \
                .assign(dtype='bool',
                        missing=df.apply(lambda x: len(x) - x.count()),
                        missing_rate=df.apply(lambda x: (len(x) - x.count()) / len(x)))
        else:
            temp = df.select_dtypes(exclude=['bool', 'number', 'object', 'datetime']).describe(include=('all')).T \
                .assign(dtype=dt,
                        missing=df.apply(lambda x: len(x) - x.count()),
                        missing_rate=df.apply(lambda x: (len(x) - x.count()) / len(x)))
        des_df.append(temp)
    des_df.reanme(columns={'pct_5': '5%', 'pct_25': '25%', 'pct_50': '50%', 'pct_75': '75%', 'pct_95': '95%'},
                  inplace=True)
    return des_df


def split_df(df, column=None, pct=0.3):
    """
    切割DataFrame
    """
    df_dict = {}
    if column is None:
        train, test = train_test_split(df, test_size=pct)
        df_dict['train'] = train
        df_dict['test'] = test
    else:
        df[column] = df[column].fillna('None_Type')
        for i in df[column]:
            df_dict[i] = df.loc[df[column] == i]
    return df_dict


def cal_vif(df, vif_columns):
    """
    计算VIF
    """
    vif_df = df.loc[:, vif_columns].fillna(-999)
    columns = vif_df.columns.tolist()
    vif_ma = vif_df.as_matrix()
    result = {}
    for k, v in enumerate(columns):
        result[v] = vif(vif_ma, k)
    vif_result = pd.Series(result, name='vif')
    vif_result.index.name = 'variable'
    vif_result = vif_result.reset_index()
    return (vif_result)


def map_level(df, dct, level=0):
    df_1 = df.copy()
    df_1.index.set_levels([[dct.get(item, item) for item in names] if i == level else names
                           for i, names in enumerate(df_1.index.levels)], inplace=True)
    return df_1


def greater_than(a, b):
    """
    函数矢量化转换
    vfunc = np.vectorize(greaterThan)
    vfunc([1,2,3,4,5],2)
    array([0, 0, 1, 1, 1])
    """
    return 1 if a > b else 0


def gen_quantile(grp_cnt):
    """
    生成等分的百分数
    gen_pro_quantile(5)
    out ：['20%', '40%', '60%', '80%', '100%']
    """
    piece = int(100 / grp_cnt)
    i = piece
    quantile = []
    while i <= 100:
        quantile.append(str(i) + '%')
        i += piece
    return quantile


def run_time(func):
    """
    函数运行时间装饰器
    """

    def wrapper(*args, **kwargs):
        import time
        t1 = time.time()
        func(*args, **kwargs)
        t2 = time.time()
        print("%s run time: %.5f s") % (func.__name__, t2 - t1)

    return wrapper


def default_param_decorator(func, **kwargs):
    """
    默认参数装饰器，给带有默认参数的函数和方法，修改其默认参数的值
    pd.DataFrame.merge_cl = defaultParamDecorator(pd.DataFrame.merge,on='contract_id',how='left')
    """
    ex_kwargs = kwargs

    def func_param(*args, **kwargs):
        return func(*args, **ex_kwargs, **kwargs)

    return func_param
