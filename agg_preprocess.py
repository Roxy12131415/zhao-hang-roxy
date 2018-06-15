"""
    功能：
    1、处理客户个人属性数据和app登录日志数据，将单列数据拆解成多列
    2、
"""
import pandas as pd

def disassemble_data(file_name,str):
    file_path = 'data/train/' + file_name + '.csv'
    data = pd.read_csv(file_path)
    columns = str.split('\t')
    data[columns] = data[str].apply(lambda x: pd.Series([i for i in x.split()]))
    del data[str]
    save_path = 'processed/' + file_name + '.csv'
    file= data.to_csv(save_path,index=None)
    return file

agg_str = 'V1	V2	V3	V4	V5	V6	V7	V8	V9	V10	V11	V12	V13	V14	V15	V16	V17	V18	V19	' \
      'V20	V21	V22	V23	V24	V25	V26	V27	V28	V29	V30	USRID'
log_str = 'USRID	EVT_LBL	OCC_TIM	TCH_TYP'

file1 = disassemble_data('train_agg',agg_str)
file2 = disassemble_data('train_log',log_str)
