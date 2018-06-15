import numpy as np
import pandas as pd
def log_tabel(data):
    #分割出第几天和时间
    data['hour'] = data.OCC_TIM.map(lambda x:x.hour)
    data['day'] = data.OCC_TIM.map(lambda x:x.day)
    #按照用户编号，统计每个用户的点击模块行为次数
    EVT_LBL_len = data.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_len':len})
    #统计每个用户点击模块种类数 set 用来得到一个集合不包含重复项
    EVT_LBL_set_len = data.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'EVT_LBL_set_len':lambda x:len(set(x))})
    #统计不同用户活跃天数
    day_set_len = data.groupby(by= ['USRID'], as_index = False)['day'].agg({'day_sum':lambda x:len(set(x))})
    #统计用户交互最多的一天的交互次数
    sort_buy_day=data.groupby(by= ['USRID','day'], as_index = False).count()
    user_maxday_count=sort_buy_day.groupby(by= ['USRID'], as_index = False)['EVT_LBL'].agg({'user_maxday_count':lambda x:max(set(x))})
    #统计每个用户这一个月第一次交互到最后一次交互的间隔天数
    Number_of_day_interval = data.groupby(by= ['USRID'], as_index = False)['day'].agg({'day_interval':lambda x:max(x)-min(x)})
    #最后一次交互距离的天数
    last_day_interval =data.groupby(by= ['USRID'], as_index = False)['day'].agg({'day_interval':lambda x:31-max(x)})
    
    return EVT_LBL_len,EVT_LBL_set_len,day_set_len,user_maxday_count,day_set_len,last_day_interval
	
	
	if __name__ == '__main__':
    train_agg = pd.read_csv('train/train_agg.csv',sep='\t')
    train_flg = pd.read_csv('train/train_flg.csv',sep='\t')
    train_log = pd.read_csv('train/train_log.csv',sep='\t',parse_dates = ['OCC_TIM'])#parse_dates:这是指定含有时间数据信息的列。
    
    all_train = pd.merge(train_flg,train_agg,on=['USRID'],how='left')
    EVT_LBL_len,EVT_LBL_set_len,day_set_len,user_maxday_count,day_set_len,last_day_interval = log_tabel(train_log)
	
	 all_train = pd.merge(all_train,EVT_LBL_len,on=['USRID'],how='left')
     all_train = pd.merge(all_train,EVT_LBL_set_len,on=['USRID'],how='left')
	 