#导入模块，导入数据，处理时间--train_log
import numpy as np
import pandas as pd
#from pandas import DataFrame
#from collections import Counter
train_log = pd.read_csv('train/train_log.csv',sep='\t',parse_dates = ['OCC_TIM'])#parse_dates:这是指定含有时间数据信息的列。
train_log['hour'] = train_log.OCC_TIM.map(lambda x:x.hour)
train_log['day'] = train_log.OCC_TIM.map(lambda x:x.day)
#train_log.head(10)#显示excel表格

#将train_log 中  EVT_LB--拆分成了三个模块加在后面--保存成df3
df2=train_log["EVT_LBL"].str.split('-', expand=True)#给EVT_LBL字段分开成三个，expand=True参数将字符串拆分成多列，返回一个数据框
df3=train_log.drop('EVT_LBL', axis=1).join(df2)#原表格按照列方向，删除EVT_LBL，然后将后来的新的一列表格df2水平连接
#df3=train_log.join(df2)#暂时不删除EVT_LBL
df3.rename(columns={0: 'EVT_LBL_1', 1: 'EVT_LBL_2', 2: 'EVT_LBL_3'}, inplace=True)#将上面列名称0，1，2改成新的名称
#df3.head(10)

# 得到每个用户在模块1-2-3中的最常用的模块（多个频繁模块则取平均）
df3['module_1']=np.array(df3['EVT_LBL_1'],dtype='int64')
df3['module_2']=np.array(df3['EVT_LBL_2'],dtype='int64')
df3['module_3']=np.array(df3['EVT_LBL_3'],dtype='int64')
EVT_LBL_1_module=df3.groupby('USRID', as_index = False)['module_1'].agg({'EVT_LBL_1_module':lambda x: np.mean(pd.Series.mode(x))})
EVT_LBL_2_module=df3.groupby('USRID', as_index = False)['module_2'].agg({'EVT_LBL_2_module':lambda x: np.mean(pd.Series.mode(x))})
EVT_LBL_3_module=df3.groupby('USRID', as_index = False)['module_3'].agg({'EVT_LBL_3_module':lambda x: np.mean(pd.Series.mode(x))})