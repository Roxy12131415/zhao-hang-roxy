#coding:utf-8

"""
    @author:Yeah!
    数据：agg个人属性数据，app操作log数据，是否有购买优惠券的flag数据
    模型：lightgbm，xgboost，lightgbm+svm,GBDT+LR,catboost,NN
"""

import pandas as pd
import numpy as np

#############################################
#############   提取log信息     ##############
#############################################
def extract_info_from_log(log_data):
    """
        log里面一共有四列数据：用户ID，模块之间的跳转信息，app操作时间，事件类型
        ① app操作时间是强特：根据app操作时间挖掘的特征有
                                   用户app交互时间差
                                   app访问路径种类数
                                   用户总交互次数
                                   最后一天的app交互次数
                                   最后三天的app交互次数
                                   最后一周的app交互次数
                                   最后3周的app交互次数
                                   用户交互天数
                                   用户交互最多的一天的交互次数
                                   第一次交互到最后一次交互的持续时间
                                   最后一次交互距离的天数
                                 （在 luoda888/2018-IJCAI-top3 中的convert_data中的trick）
        ② 模块之间的跳转数据
        ③ 事件类型也据说是强特

    :param log_data:直接输入log中的原数据
    :return:
    """log['OCC_TIM'] = log['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
log = log.sort_values(['USRID','OCC_TIM'])
log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)

log = log.groupby(['USRID'],as_index=False)['next_time'].agg({
    'next_time_mean':np.mean,
    'next_time_std':np.std,
    'next_time_min':np.min,
    'next_time_max':np.max
})
    # 用户app交互时间差引申的特征
    log = pd.
    log['OCC_TIM_timestamp'] = log_data['OCC_TIM'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
    log = log.sort_values(['USRID', 'OCC_TIM_timestamp'])
    log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)

    OCC_TIM_diff = log_data.groupby(['USRID'], as_index=False)['next_time'].agg({
        'next_time_mean': np.mean,
        'next_time_std': np.std,
        'next_time_min': np.min,
        'next_time_max': np.max
    })


    # app访问路径种类数
    EVT_LBL_set_len = log_data.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg({'EVT_LBL_set_len': lambda x: len(set(x))})  # app访问路径的种类数

    # 用户总交互次数
    EVT_LBL_len = log_data.groupby(by=['USRID'], as_index=False)['EVT_LBL'].agg({'EVT_LBL_len': len})


    log_data['hour'] = log_data.OCC_TIM.map(lambda x: x.hour)
    log_data['day'] = log_data.OCC_TIM.map(lambda x: x.day)
    info_from_log
    return info_from_log



#############################################
#############   加载训练数据    ##############
#############################################

train_agg = pd.read_csv('data/train_agg.csv',sep='\t')
train_flg = pd.read_csv('data/train_flg.csv',sep='\t')
train_log = pd.read_csv('data/train_log.csv',sep='\t',parse_dates=['OCC_TIM']) #参数parse_dates后面传入要解析的列，将列解析为日期格式

all_train = pd.merge(train_flg,train_agg,how='left',on=['USRID']) # 之后我们要在该训练数据的基础上，再提取log中的特征添加进来





















# 读取个人信息
train_agg = pd.read_csv('data/train_agg.csv',sep='\t')
test_agg = pd.read_csv('data/test_agg.csv',sep='\t')
agg = pd.concat([train_agg,test_agg],copy=False)

# 日志信息
train_log = pd.read_csv('data/train_log.csv',sep='\t')
test_log = pd.read_csv('data/test_log.csv',sep='\t')
log = pd.concat([train_log,test_log],copy=False)

log['EVT_LBL_1'] = log['EVT_LBL'].apply(lambda x:x.split('-')[0])
log['EVT_LBL_2'] = log['EVT_LBL'].apply(lambda x:x.split('-')[1])
log['EVT_LBL_3'] = log['EVT_LBL'].apply(lambda x:x.split('-')[2])

# 用户唯一标识
train_flg = pd.read_csv('data/train_flg.csv',sep='\t')
test_flg = pd.read_csv('data/submit_sample.csv',sep='\t')
test_flg['FLAG'] = -1
del test_flg['RST']
flg = pd.concat([train_flg,test_flg],copy=False)

data = pd.merge(agg,flg,on=['USRID'],how='left',copy=False)

import time

# 这个部分将时间转化为秒，之后计算用户下一次的时间差特征
# 这个部分可以发掘的特征其实很多很多很多很多
log['OCC_TIM'] = log['OCC_TIM'].apply(lambda x:time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S")))
log = log.sort_values(['USRID','OCC_TIM'])
log['next_time'] = log.groupby(['USRID'])['OCC_TIM'].diff(-1).apply(np.abs)

log_2 = pd.DataFrame()
log_2 = log.groupby(['USRID'],as_index=False)['next_time'].agg({
    'next_time_mean':np.mean,
    'next_time_std':np.std,
    'next_time_min':np.min,
    'next_time_max':np.max
})

# 对于log模块，加入app使用次数信息
# log_3 = log.groupby('USRID',as_index=False)['OCC_TIM'].count()
# log_3.reset_index(drop=True)
#
#
# cols_to_use = log_3.columns.difference(log_2.columns)
# log_4 = pd.merge(log_2,log_3[cols_to_use],left_index=True,right_index=True,how='outer')

data2 = pd.merge(data,log_2,on=['USRID'],how='left',copy=False)


moduel1 = log.pivot_table('OCC_TIM',index='USRID',columns='EVT_LBL_1',aggfunc=len)
moduel1.fillna(0,inplace=True)
moduel1.reset_index(drop=True)

cols_to_use2 = moduel1.columns.difference(data2.columns)
data = pd.merge(data2,moduel1[cols_to_use2],left_index=True,right_index=True,how='outer')

from sklearn.model_selection import StratifiedKFold

train = data[data['FLAG']!=-1]
test = data[data['FLAG']==-1]

train_userid = train.pop('USRID')
y = train.pop('FLAG')
col = train.columns
X = train[col].values

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test = test[col].values

N = 5
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=42)

import lightgbm as lgb
from sklearn.metrics import roc_auc_score

xx_cv = []
xx_pre = []

for train_in,test_in in skf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]

    # create dataset for lightgbm
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # specify your configurations as a dict
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 32,
        'learning_rate': 0.01,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=40000,
                    valid_sets=lgb_eval,
                    verbose_eval=250,
                    early_stopping_rounds=50)

    # print('Save model...')
    # save model to file
    # gbm.save_model('model.txt')

    print('Start predicting...')
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    xx_cv.append(roc_auc_score(y_test,y_pred))
    xx_pre.append(gbm.predict(test, num_iteration=gbm.best_iteration))

s = 0
for i in xx_pre:
    s = s + i

s = s /N

res = pd.DataFrame()
res['USRID'] = list(test_userid.values)
res['RST'] = list(s)

print('xx_cv',np.mean(xx_cv))

import time
time_date = time.strftime('%Y-%m-%d',time.localtime(time.time()))
res.to_csv('submit/%s_%s.csv'%(str(time_date),str(np.mean(xx_cv)).split('.')[1]),index=False,sep='\t')
