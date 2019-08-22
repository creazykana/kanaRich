# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:00:59 2018

@author: fanhg
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import feature_engineer as fe
import modeler as md
import xgboost as xgb
import data_processing as dp
import os

os.chdir('E:\业务文档备份\python_code\评分卡模型\code')

'''
    特征工程1.连续变量
'''
df = pd.read_pickle('../model_data/建模样本.pkl')
use_merge = df.copy()

num_variables = ['valid_time', 'date_fisrt_lic_time', 'operation_time', 'rent_time', 'biz_area',
                 'registered_capital', 'amt_2016_oneyear', 'amt_2017_halfyear', 'amt_total', 'amt_201601',
                 'amt_201602', 'amt_201603', 'amt_201604', 'amt_201605', 'amt_201606', 'amt_201607',
                 'amt_201608', 'amt_201609', 'amt_201610', 'amt_201611', 'amt_201612', 'amt_201701',
                 'amt_201702', 'amt_201703', 'amt_201704', 'amt_201705', 'amt_201706', 'qty_demand_total',
                 'qty_order_total', 'demand_16', 'demand_17', 'qty_order_2016_oneyear', 'qty_order_2017_halfyear',
                 'max_amt_sum', 'supplyrate_total', 'supplyrate_aver', 'supplyrate_16', 'supplyrate_17', 'aver_amt',
                 'aver_amt_all_months', 'amt_stddev', 'max_aver', 'rate_9_8', 'period_median',
                 'period_mean', 'operate_total_days', 'suspend_count', 'suspend_max', 'dif_months',
                 'dif_months_17', 'total_fre_count', 'telphone_num',
                 'name_num', 'mobilephone_num', 'address_num', 'period_order_sum_std', 'period_order_sum_avg',
                 'period_avg', 'order_sum_per_day', 'period_cv', 'month_sum_cv', 'last_order_date_diff',
                 'order_time_length',
                 'operator_length', 'quart_growth', 'month_growth', 'max_month_diff', 'max_down_velocity', 'max_down',
                 'legal_person_age']
cate_variables = ['retail_type', 'market_type', 'work_state', 'work_state_name', 'legal_person_sex',
                  'province', 'city', 'v_province_name', 'same_prov', 'same_city', 'lic_first_year',
                  'now_effect_status', 'now_effect_status_translate',
                  'manager_scope', 'manager_scope_01', 'manager_scope_02', 'manager_scope_04',
                  'ground_ownership_translate',
                  'retail_cid_type', 'special_person_group',
                  'special_person_group01', 'special_person_group08', 'area_type', 'eco_type', 'eco_type_translate',
                  'area_range', 'if_is_chain',
                  'if_is_main', 'last_operate_type', 'last_operate_type_translate', 'average1617', 'recent_1m',
                  'recent_2m',
                  'recent_3m', 'rank_amt_total_5', 'rank_amt_total_retail_5', 'rank_amt_total_market_5',
                  'rank_amt_total_vprovince_name_5',
                  'rank_amt2016_5', 'rank_amt2016_5', 'rank_amt_17_16_change', 'match_type', 'is_tobacco_lic_valid',
                  'is_retail_validate',
                  'is_matched', 'is_lincense_expired', 'is_business_lic_valid', 'has_telphone', 'has_mobilephone']

# 相关系数
'''
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
'''
use_merge_dropcorr = dp.drop_corr(use_merge[num_variables], threshold=0.9)  # 删除连续变量中相关系数高的字段
corr_cols = use_merge_dropcorr.columns.tolist()  # 获取剩下的字段名称

# 决策树分箱
# 4.决策树分箱，根据iv剔除变量
'''
def map_bin(x, bins):#bins为决策树运行出来的阙值数
    kwargs = {}
    if x == max(bins):
        kwargs['right'] = True #若x达到上限值，令变量kwargs的'right'为True
    the_bin = bins[np.digitize([x], bins, **kwargs)[0]]
    bin_lower = bins[np.digitize([x], bins, **kwargs)[0] - 1]
    return '{0}~{1}'.format(round(bin_lower, 4), round(the_bin, 4))

class FeatureBinning(object):
    def __init__(self, bin_type=2, criterion='entropy', max_leaf_nodes=8,
                 min_samples_leaf=100, max_depth=4, bin_count=20, na=-999, bins_dict=None):
        """
        bin_type:{1:等分分箱，2：决策数分箱,3,线性拟合分箱，4:分类变量进行分箱合并，5:分类变量不分箱,6：手动分箱}
        """
        self.na = na
        self.bin_type = bin_type
        self.criterion = criterion
        self.max_leaf_nodes = max_leaf_nodes
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.bin_count = bin_count
        self.bins_dict = bins_dict
    def decision_split(self, var, y):#决策树分箱
        min_samples_leaf = min(len(y) / 10, self.min_samples_leaf)#最小的叶数，min(样本数/10，设定值900)
        dt = DecisionTreeClassifier(criterion=self.criterion,#信息增益
                                    max_depth=self.max_depth,#最大深度4
                                    max_leaf_nodes=self.max_leaf_nodes,#最大叶支点数8
                                    min_samples_leaf=min_samples_leaf)#最小叶片数
        var2 = var.values.reshape(-1, 1)#将要分箱的变量转换成n*1的矩阵
        dt.fit(var2, y)#训练模型
        bins = np.unique(dt.tree_.threshold[dt.tree_.feature > -2])#把特征大于-2的阙值数提出来
        if len(bins) > 0:#若成功分组
            bins[0] = -np.inf#定义下限为负无穷
            bins[-1] = np.inf#定义上限为正无穷
            bucket = var.apply(map_bin, bins=bins)#调用函数map_bin
        else:
            bucket = None
        return bucket
    def binning_series(self, var, y, bin_type=None, bin_count=None):
        #var=df[i],y=df[label]
        bin_type = bin_type if bin_type in [1, 2, 3, 4, 5, 6] else self.bin_type#若赋给的bin_type值在1~6内则正常赋值，若非则以对象变量bin_type赋值(默认2)
        bin_count = bin_count if bin_count else self.bin_count#同上，默认20
        var1 = var.fillna(self.na)#对缺失值进行赋值，self_na = -999
        if bin_type == 1:
            bucket = self.equal_split(var1, y, bin_count=bin_count)
        elif bin_type == 2:#调用决策树分箱
            bucket = self.decision_split(var1, y)
            if bucket is None:#若分享的结果是空，则调用no_split即bucket = var1
                bucket = self.no_split(var1)
            else:
                bucket
        elif bin_type == 3:
            bucket = self.line_split(var1, y, bin_count=bin_count)
        elif bin_type == 4:
            bucket = self.cate_split(var1, y)
        elif bin_type == 5:
            bucket = self.no_split(var1)
        elif bin_type == 6:
            bucket = self.hand_split(var1, y)
        else:
            print(u'已没有其他选项')
        return bucket

    def binning_df(self, df, binning_feature, label, bin_type=None, bin_count=None):
        #df=use_merge,binning_feature=corr_cols,label='target'
        df_columns = df.columns.tolist()#提取原始数据文件的字段名称
        df_columns.remove(label)#删除目标变量字段
        in_columns = [i for i in df_columns if i in binning_feature]#挑选出在binning_feature中的字段名
        out_columns = [i for i in binning_feature if i not in df_columns]#挑选出不在binning_feature中的字段名
        if len(out_columns) > 0:
            print(out_columns, 'not in dataframe!')#如果不在binning_feature中的字段的个数大于零，打印出字段的名称
        for i in in_columns:
            bucket = self.binning_series(df[i], df[label], bin_type=bin_type, bin_count=bin_count)#对每个字段调用函数binning_series
            df[i + '_bin'] = bucket#创建新字段：原字段名 + '_bin'
        return df
'''
bin_df = fe.FeatureBinning(min_samples_leaf=900, max_depth=4).binning_df(use_merge, corr_cols, 'target')
bin_cols = [i for i in bin_df if i.endswith('bin')]
'''
def feature_stats_all(df, all_feature, label, bad_weight=1, good_weight=1,datatype='num',qushi='up'):
    """
    对分箱变量进行统计
    df:输入分好箱的DataFrame
    all_feature:分好箱的所有列名的list
    label:目标变量
    """
    df_columns = df.columns.tolist() #获取列名
    df_columns.remove(label) #删掉目标变量列
    in_columns = [i for i in df_columns if i in all_feature] #取出分箱字段
    out_columns = [i for i in all_feature if i not in df_columns] #取出其他数据字段
    if len(out_columns) > 0:
        print(out_columns, 'not in dataframe!')
    stats_all_df = pd.DataFrame()
    for i in in_columns:
        feature_stats = FeatureStats(df[i], df[label],
                                     bad_weight=bad_weight,
                                     good_weight=good_weight,datatype=datatype).stats_feature()
        stats_all_df = stats_all_df.append(feature_stats) 
    return stats_all_df
'''
stats_df = fe.feature_stats_all(bin_df, bin_cols, 'target')  # 输出字段的woe、iv、gini系数
stats_left = stats_df.loc[stats_df.iv >= 0.01]
var_left = stats_left['var'].unique().tolist()  # 保留特征 29个
feature_left = [i.split('_bin')[0] for i in var_left]

# 随机森林
use_merge_fillna = df.copy()
use_merge_fillna[num_variables] = use_merge_fillna[num_variables].fillna(use_merge_fillna.mean())  # 连续变量用均值填充缺失值
stable_df = fe.stable_select(use_merge_fillna, 'target', corr_cols, model='rfr')  # 随机森林回归
stable_left = stable_df.head(30).index.tolist()  # 保留特征 30个
all_left_cols = list(set(feature_left) | set(stable_left))  # 决策树分箱和随机森林共保留特征 32个

# vif
use_merge_dropvif = dp.cal_vif(use_merge, all_left_cols)  # 计算vif值
vif_cols = use_merge_dropvif.loc[use_merge_dropvif['vif'] < 500, 'variable'].tolist()  # 32

"""
    特征分箱的调整
"""
# 1.连续变量
hand_bin_cols = [i + '_bin' for i in vif_cols]  # 给每个字段名后面加上_bin
stats_hand_df = stats_df.loc[stats_df['var'].isin(hand_bin_cols)]  # 选择性输出数据
stats_hand_df['variable'] = [x + str(y) for x, y in zip(stats_hand_df['var'], stats_hand_df.index)]  # 字段名+编号(index)
# stats_hand_df.to_csv('../result/woe_decision_tree_df.csv')

use_merge_fill_value = df.copy()
use_merge_fill_value['registered_capital'] = np.where(use_merge_fill_value['registered_capital'] > 999999, \
                                                      np.nan, use_merge_fill_value['registered_capital'])
use_merge_fill_value[num_variables] = use_merge_fill_value[num_variables].fillna(-999)

# 手动分箱
num_bins_dict = {'valid_time': [-99999999, -999, 57.371, 99999999],
                 'date_fisrt_lic_time': [-99999999, -999, 52.9516, 99999999],
                 'operation_time': [-99999999, -999, 15.7581, 99999999],
                 'biz_area': [-99999999, -999, 22.47, 53.92, 99999999],
                 'registered_capital': [-99999999, -999, 0.375, 99999999],
                 'amt_2016_oneyear': [-99999999, -999, 104167.75, 375194.125, 99999999],
                 'amt_2017_halfyear': [-99999999, -999, 99903.1016, 99999999],
                 'amt_201601': [-99999999, -999, 27825.5703, 99999999],
                 'amt_201602': [-99999999, -999, 8308.8105, 24846.9297, 99999999],
                 'amt_201612': [-99999999, -999, 4720.9751, 20570.3457, 99999999],
                 'amt_201704': [-99999999, -999, 5335.1553, 15868.4805, 99999999],
                 'qty_demand_total': [-99999999, -999, 90.1815, 99999999],
                 'supplyrate_aver': [-99999999, -999, 4343.7529, 5768.7354, 99999999],
                 'rate_9_8': [-99999999, -999, 0.1355, 99999999],
                 'operate_total_days': [-99999999, -999, 499.5, 531.5, 99999999],
                 '_17_order_fre_count': [-99999999, -999, 22.5, 99999999],
                 'legal_person_age': [-99999999, -999, 35.5, 41.5, 99999999],
                 'operator_length': [-99999999, -999, 1.311, 1.5877, 99999999],
                 'quart_growth': [-99999999, -999, -6.2202, 99999999],
                 'max_down': [-99999999, -999, -72.5897, 99999999]
                 }

# z_auto_bin = fe.feature_stats_series(use_merge, 'id1', 'npd30', bin_type=2, bins_dict=num_bins_dict, min_samples_leaf=860)
# z_hand_bin = fe.feature_stats_series(use_merge_fill_value, 'id285', 'npd30', bin_type=6, num_bins_dict=bins_dict1)

hand_bin_df = fe.FeatureBinning(bin_type=6, bins_dict=num_bins_dict).binning_df(use_merge_fill_value,
                                                                                list(num_bins_dict.keys()),
                                                                                'target')  # 手动分箱
hand_bin_df = hand_bin_df.dropna(how='all', axis=1)  # 删除均为空的数据
hand_bin_cols = [i for i in hand_bin_df.columns if i.endswith('bin')]  # 挑出以'bin'为结尾的字段名

# 分布与Badrate
woe_stats_df = fe.feature_stats_all(hand_bin_df, hand_bin_cols, 'target')  # 输出字段的woe、iv、gini系数
pct_cols = [i.split('_bin')[0] for i in hand_bin_cols]  # 提取_bin前面的字段
woe_stats_df['variable'] = [x + str(y) for x, y in zip(woe_stats_df['var'], woe_stats_df.index)]  # 字段后面加编号

import importlib as imp
import modeler as md

imp.reload(md)  # 重新加载模块md,若md模块发生改变时会有很大的作用
md.deploy_lr_df(woe_stats_df, 'variable_distribution.py', 'variable')
import variable_distribution

imp.reload(variable_distribution)
from variable_distribution import *

use_merge_dist = use_merge_fill_value.copy()
for i in pct_cols:  # 对每个字段调用函数 字段名_map ，函数存再包variable_distribution内
    use_merge_dist[i + '_bucket'] = use_merge_dist.apply(eval(i + '_map'), axis=1)

train_valid_df, test_df = train_test_split(use_merge_dist, test_size=0.2)  # 把数据拆分成训练集和测试集，测试集的比例是0.2
train_df, valid_df = train_test_split(train_valid_df, test_size=0.3)
train_df['tag'] = 'train'
valid_df['tag'] = 'valid'
test_df['tag'] = 'test'
use_merge_dist = pd.concat([train_df, valid_df, test_df])

print('BadRate of Training Set: {:.3%}'.format(train_df['target'].mean()))
print('BadRate of Valid Set: {:.3%}'.format(valid_df['target'].mean()))
print('BadRate of Test Set: {:.3%}'.format(test_df['target'].mean()))

tag = pd.unique(use_merge_dist.tag.sort_values())  # ['test', 'train', 'valid']
stats_cols = ['Bad' + str(i) for i in tag] + ['Bad'] + ['Total' + str(i) for i in tag] + ['Total']  # 定义几个字段名
badrate_cols = ['Badrate' + str(i) for i in tag] + ['Badrate']
pct_cols = ['Pct' + str(i) for i in tag] + ['Pct']

table_distr = pd.DataFrame()
bucket_cols = [i for i in use_merge_dist.columns if i.endswith('bucket')]  # 提取表use_merge_dist中以bucket结尾的字段名，224行出现错误
for i in bucket_cols:
    table_stats = pd.pivot_table(use_merge_dist, values='target', index=[i],
                                 columns=['tag'], aggfunc=[np.sum, len], margins=True)
    table_badrate = table_stats['sum'] / table_stats['len']
    table_stats.index.name = None;
    table_badrate.index.name = None
    table_stats.columns = stats_cols
    table_stats_badrate = pd.concat([table_stats, table_badrate], ignore_index=True, axis=1)
    table_stats_badrate.columns = stats_cols + badrate_cols
    total_cols = [i for i in table_stats_badrate.columns if i.startswith('Total')]
    table_pct = table_stats_badrate.loc[:, total_cols] / table_stats_badrate.loc['All', total_cols]
    table_stats_badrate_pct = pd.concat([table_stats_badrate, table_pct], ignore_index=True, axis=1)
    table_stats_badrate_pct.columns = stats_cols + badrate_cols + pct_cols
    table_distr = pd.concat([table_distr, table_stats_badrate_pct])

table_distr.index.name = 'variable'
table_distr_sec = table_distr.reset_index()
table_distr_thi = woe_stats_df.loc[:, ['variable', 'bin_group', 'iv']].merge(table_distr_sec, on='variable', how='left')
# table_distr_thi.to_csv('../result/table_distr_thi.csv')

# 离散变量
use_merge_fill_value[cate_variables].dtypes  # 离散变量字段类型
cate_df = fe.FeatureBinning(bin_type=5).binning_df(use_merge_fill_value, cate_variables, label='target',
                                                   bin_type=5)  # 对离散变量的值进行分类不分箱
cate_bin_cols = [i for i in cate_df if i.endswith('bin') and i not in hand_bin_cols and  # 挑选分类后的离散变量名
                 i not in ['date_rent_end_bin', 'date_rent_start_bin']]
cate_stats_df = fe.feature_stats_all(cate_df, cate_bin_cols, 'target', datatype='non-num')  # 计算各变量的woe,iv值
# cate_stats_df.to_excel('../result/cate_variable_woe.xlsx')

use_merge_cate_df = use_merge_fill_value.copy()
cate_bins_dict = {'retail_type': [['乡村'], ['城镇']],
                  'ground_ownership_translate': [['租赁（长期）', 'null'], ['自有', '无偿使用（长期）']],
                  'is_business_lic_valid': [['1'], ['0']],
                  'is_retail_validate': [['1'], ['0']],
                  'last_operate_type_translate': [['责令停业', '歇业', '新办', '审批注销'],
                                                  ['变更', '停业', 'null', '延续', '恢复营业', '登记注销', '责令停业恢复营业']],
                  'legal_person_sex': [['男'], ['女'], ['null']],
                  'manager_scope_02': [['1'], ['0']],
                  'manager_scope_04': [['1'], ['0']],
                  'market_type': [['烟酒商店', '便利店', '超市'], ['娱乐服务类', '食杂店', '其他', '商场']],
                  'match_type': [['1181', '1191', '1111', '0'], ['1110'],
                                 ['110', '111', '101', '1101', '1180', '1190', '181', '191', '180']],
                  'now_effect_status_translate': [['没有解释', '责令停业', '注销'], ['null', '正常经营', '暂停营业'], ['初始申请']],
                  'province': [['浙江省', '福建省', '内蒙古自治区'], ['新疆维吾尔自治区', '广东省', '甘肃省', '湖南省', '河南省'],
                               ['北京市', '江苏省', '山西省', '辽宁省', '重庆市', '湖北省', '江西省', '河北省'],
                               ['四川省', '山东省', '贵州省', '天津市', '云南省', '广西壮族自治区',
                                '宁夏回族自治区', '黑龙江省', '吉林省', '海南省', '西藏自治区', '上海市']],
                  'rank_amt_17_16_change': [['下降'], ['上升'], ['不变']],
                  'work_state_name': [['失效', '正常客户', '无效', '11', '注销', '正常', '停用', '暂停'], ['02', '0', '正常营业', '启用'],
                                      ['延续', '有效', '停业（强制歇业）', '停业', '歇业', 'null', '06', '04', '取消', '营业']]
                  }
cate_bin_df = fe.FeatureBinning(bin_type=4, bins_dict=cate_bins_dict).binning_df(use_merge_cate_df,
                                                                                 list(cate_bins_dict.keys()), 'target',
                                                                                 bin_type=4)  # 4:分类变量进行分箱合并
cate_df_cols = [i for i in cate_bins_dict]
cate_df_bin_cols = [i + '_bin' for i in cate_df_cols]
cate_woe_stats_df = fe.feature_stats_all(cate_bin_df, cate_df_bin_cols, 'target', datatype='non-num')  # 计算变量的ks,woe,iv值

# map_woe
'''
def map_woe(bin_df, woe_map, model_bin_columns):
    """
    将分好箱的Dataframe进行WOE Map
    bin_df：使用feature_bin_hand方法分好箱的Dataframe
    woe_map: feature_stats_all统计好的分箱对应
    model_bin_columns: 分好箱的字段
    """
    model_stats_left = woe_map.loc[:, ['var', 'bin_group', 'woe']] #选取三列字段的数据
    woe_dict = {}
    for i in model_stats_left['var'].unique(): #对特征名称做循环，如valid_time_bin
        temp = model_stats_left.loc[model_stats_left['var'] == i][['bin_group', 'woe']] #把特征名对应的bin_group和woe字段拿出来
        temp1 = temp.set_index('bin_group')
        woe_dict[i] = temp1.to_dict()['woe']
    for i in model_bin_columns:
        bin_df[i + '_woe'] = bin_df[i].map(woe_dict[i])
    return bin_df
'''
num_woe_df = fe.map_woe(hand_bin_df, woe_stats_df, hand_bin_cols)
# 连续变量Woe
# woe_stats_df.to_excel('../result/连续变量Woe值.xlsx')

cate_bin_df = fe.FeatureBinning(bin_type=4, bins_dict=cate_bins_dict).binning_df(num_woe_df,
                                                                                 list(cate_bins_dict.keys()), 'target',
                                                                                 bin_type=4)
cate_df_cols = [i for i in cate_bins_dict]
cate_df_bin_cols = [i + '_bin' for i in cate_df_cols]
cate_woe_stats_df = fe.feature_stats_all(cate_bin_df, cate_df_bin_cols, 'target', datatype='non-num')

cate_woe_df = fe.map_woe(cate_bin_df, cate_woe_stats_df, cate_df_bin_cols)
# 离散变量Woe
# cate_woe_stats_df.to_excel('../result/离散变量Woe值.xlsx')
num_cate_woe_df = cate_woe_df.copy()

woe_cols = [i for i in num_cate_woe_df.columns if i.endswith('woe')]

# 逐步回归训练模型
stepwise_x = num_cate_woe_df[woe_cols]
stepwise_y = num_cate_woe_df['target']
stepwise_x_left = dp.drop_corr(stepwise_x[woe_cols], threshold=0.7)
stepwise_x_left['operate_total_days_bin_woe'] = np.where(np.isinf(stepwise_x_left['operate_total_days_bin_woe']),
                                                         0.2931, stepwise_x_left['operate_total_days_bin_woe'])
stepwise_x_corr_cols = stepwise_x_left.columns.tolist()

stepwise_result = md.StepwiseModel(stepwise_x_left, stepwise_y).stepwise()
stepwise_cols = [i for i in stepwise_result if stepwise_result[i] < 0 and i not in ['const']]
stepwise_source_cols = [i.split('_bin')[0] for i in stepwise_cols]

# 建模woe数据
model_df_woe = stepwise_x_left[stepwise_cols].copy()
model_df_woe['target'] = stepwise_y
# model_df_woe.to_pickle('../model_data/建模WoE数据.pkl')


train_valid_df, test_df = train_test_split(model_df_woe, test_size=0.2)
train_df, valid_df = train_test_split(train_valid_df, test_size=0.3)
train_df_bad = train_df[train_df['target'] == 1]
train_df_good = train_df[train_df['target'] == 0].sample(n=3000)
train_df_samples = pd.concat([train_df_bad, train_df_good])
train_df_samples = train_df_samples.reset_index()
valid_df = valid_df.reset_index()
test_df = test_df.reset_index()

print(np.mean(train_df_samples.target))
print(np.mean(valid_df.target))
print(np.mean(test_df.target))

x_variables = stepwise_cols.copy()
x_variables.remove('retail_type_bin_woe')
x_variables.remove('biz_area_bin_woe')
x_variables.remove('operator_length_bin_woe')
# x_variables.remove('is_business_lic_valid_bin_woe')
# x_variables.remove('id268_bin_woe')
# x_variables.remove('id306_bin_woe')
# x_variables.remove('id303_bin_woe')
# x_variables.remove('id263_bin_woe')
# x_variables.remove('id83_bin_woe')
# x_variables.remove('id145_bin_woe')

train_param, train_predict_data = md.logit_fit(train_df_samples, 'target', x_variables)
train_plot = md.ModelPlot(train_predict_data, y_variable='target', title='Train')
train_plot.roc_plot()
train_plot.ks_plot()

valid_param, valid_predict_data = md.logit_fit(valid_df, 'target', x_variables)
valid_predict_data = md.logit_predict(valid_df, 'target', x_variables, train_param)
valid_plot = md.ModelPlot(valid_predict_data, y_variable='target', title='Valid')
valid_plot.roc_plot()
valid_plot.ks_plot()

test_param, test_predict_data = md.logit_fit(test_df, 'target', x_variables)
test_predict_data = md.logit_predict(test_df, 'target', x_variables, train_param)
test_plot = md.ModelPlot(test_predict_data, y_variable='target', title='Test')
test_plot.roc_plot()
test_plot.ks_plot()

data_dict = {'train': train_df_samples, 'valid': valid_df, 'test': test_df}
md.quick_fit_plot(data_dict, x_variables, 'target')

# 15.模型稳定性验证
beta_dict = {'const': -2.5585,
             'legal_person_age_bin': -0.8039,
             'province_bin': -0.7214,
             'legal_person_sex_bin': -0.8781,
             'supplyrate_aver_bin': -0.8203,
             '_17_order_fre_count_woe': -1.7954,
             'qty_demand_total_bin': -1.0195,
             'rank_amt_17_16_change_bine': -1.6530,
             'is_business_lic_valid_bin': -0.5652
             }

# md.deploy_lr_file('../result/连续变量Woe值-map评分.xlsx', 'continous_variables_score.py', target='score')

imp.reload(scorecard_v4)
from scorecard_v4 import *

# 建模样本映射评分

model_cols = ['legal_person_age', 'province', 'legal_person_sex', 'supplyrate_aver',
              '_17_order_fre_count', 'qty_demand_total', 'rank_amt_17_16_change', 'is_business_lic_valid']
model_samples_score = pd.read_pickle('../model_data/建模样本.pkl')
for i in model_cols:
    model_samples_score[i + '_score'] = model_samples_score.apply(eval(i + '_map'), axis=1)
model_samples_score['base_score'] = 454.25 - (-2.5585) * 40 / np.log(2)
score_cols = [i for i in model_samples_score.columns if i.endswith('_score')]
model_samples_score['final_score'] = model_samples_score[score_cols].sum(axis=1)

# count    9766.000000
# mean      626.182269
# std        58.092246
# min       455.689710
# 25%       585.770210
# 50%       630.013810
# 75%       669.978110
# max       757.253010
# Name: final_score, dtype: float64

aa = fe.feature_stats_series(model_samples_score, 'final_score', label='target', bin_count=10, bin_type=1)
score_bin_dict = {'final_score': [0, 547.5, 573.5, 594.5, 613.0, 645.0, 662.0, 697.5, 9999]}
ks_result = fe.feature_stats_series(model_samples_score, 'final_score', 'target', bin_type=6, bins_dict=score_bin_dict)

# 闭卷测试样本的ks
closed_test_df = pd.read_pickle('../model_data/闭卷测试样本.pkl')
for i in model_cols:
    closed_test_df[i + '_score'] = closed_test_df.apply(eval(i + '_map'), axis=1)
closed_test_df['base_score'] = 454.25 - (-2.5585) * 40 / np.log(2)
score_cols = [i for i in closed_test_df.columns if i.endswith('_score')]
closed_test_df['final_score'] = closed_test_df[score_cols].sum(axis=1)

# count    9766.000000
# mean      626.182269
# std        58.092246
# min       455.689710
# 25%       585.770210
# 50%       630.013810
# 75%       669.978110
# max       757.253010
# Name: final_score, dtype: float64

aa = fe.feature_stats_series(closed_test_df, 'final_score', label='target', bin_count=5, bin_type=1)
score_bin_dict = {'final_score': [0, 547.5, 573.5, 594.5, 613.0, 645.0, 662.0, 697.5, 9999]}
ks_result = fe.feature_stats_series(closed_test_df, 'final_score', 'target', bin_type=6, bins_dict=score_bin_dict)