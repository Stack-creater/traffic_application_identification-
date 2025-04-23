#%%
import os
import datetime
import json
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import haslib

ASSETS='assets/'
HS_RESOURCES='hs_resources/'

# ---读入资源---
# app_map: app_id -> app_name
with open(ASSETS+'app_map.json') as f:
    app_map=json.load(f)
    app_map={ app_id:app_name for app_name,app_id in app_map.items()}
# bow_map: bow_id->hash_code;   hash2word: hash_code->word
bow_map=pd.read_csv(ASSETS+'bow_map.txt',header=None,names=['col'])
bow_map=bow_map['col']
hash2word=pd.read_csv(ASSETS+'dict.csv',header=None,index_col=1).squeeze().drop_duplicates()
# 读入数据集
dataset=pd.read_csv(ASSETS+'training.csv')
dataset.fillna(value={'http_host':'','tls_sni':''}, inplace=True)
dataset['raw_string']=dataset['http_host']+dataset['tls_sni']
dataset['label']=dataset['label_name']
dataset.drop(['http_host','tls_sni','label_name'],axis=1,inplace=True)
#0-105列为统计特征，106列开始为bow特征，-2,-1列为raw_string,label
bow_cnt=dataset.iloc[:,106:-2]  
dataset=dataset[(bow_cnt !=0).any(axis=1)]  #去掉bow_count均为0的样本
dataset['raw_string']=dataset['raw_string'].apply(lambda x: np.where(x!='',x,None))
dataset.dropna(how='any',subset=['raw_string'],inplace=True)  #去掉raw_string (http_host+tls_sni)为空的样本

print('各app样本数:')
print(dataset['label'].value_counts())
print()

# 暂停后等待用户按下回车键
input("按下回车键继续执行：")
# 继续执行后面的代码
print("继续执行...")
# ---划分样本---
X_whole=dataset.iloc[:,:-1] #去掉label
y_whole=dataset['label'].values
X_train,X_test,y_train,y_test=train_test_split(X_whole,y_whole,test_size=0.3,random_state=0)

X_test_rawstr=X_test[['raw_string']]
X_whole,X_train,X_test=X_whole.iloc[:,:-1],X_train.iloc[:,:-1],X_test.iloc[:,:-1]#去掉raw_string
X_train_stats,X_test_stats=X_train.iloc[:,:106],X_test.iloc[:,:106]
X_train_bow,X_test_bow=X_train.iloc[:,106:] ,X_test.iloc[:,106:]


# ---训练、测试模型---
stats_rf_model=RandomForestClassifier(n_jobs=-1,random_state=0)
stats_rf_model.fit(X_train_stats,y_train)
begin=datetime.datetime.now()
score=stats_rf_model.score(X_whole.iloc[:,:106],y_whole)
total=datetime.datetime.now()-begin
print('基于stats特征的模型预测耗时',total.seconds,'s',total.microseconds,'us')
print('基于统计特征的模型预测准确率:',stats_rf_model.score(X_test_stats,y_test))

bow_rf_model=RandomForestClassifier(n_jobs=-1,random_state=0)
bow_rf_model.fit(X_train_bow,y_train)
begin=datetime.datetime.now()
score=bow_rf_model.score(X_whole.iloc[:,106:],y_whole)
total=datetime.datetime.now()-begin
print('基于bow特征的模型预测耗时',total.seconds,'s',total.microseconds,'us')
print('基于bow特征的模型预测准确率',bow_rf_model.score(X_test_bow,y_test))

rf_model=RandomForestClassifier(n_jobs=-1,random_state=0)
rf_model.fit(X_train,y_train)
begin=datetime.datetime.now()
score=rf_model.score(X_whole,y_whole)
total=datetime.datetime.now()-begin
print('基于全部特征的模型预测耗时',total.seconds,'s',total.microseconds,'us')
print('基于全部特征的模型预测准确率:',rf_model.score(X_test,y_test))
print()

# 暂停后等待用户按下回车键
input("按下回车键继续执行：")
# 继续执行后面的代码
print("继续执行...")


# ---提取importance、latency benchmark---
rawstr_dataset=dataset.iloc[:,-2:].sort_values(by='label')
rawstr_dataset.to_csv(HS_RESOURCES+'rawstr_dataset.txt',sep=' ',header=0,index=False)

model_importances=bow_rf_model.feature_importances_
classified_importances=haslib.get_class_feature_importance(X_train_bow,y_train,model_importances)
topX_list=[20]
print('---规则匹配latency benchmark---')
for x in topX_list:
    topX_importances=haslib.get_topX_importances(classified_importances,topX=x)
    rulesets=haslib.get_rulesets(topX_importances,bow_map,hash2word)
    print('topX=',x)
    # print('---hs benchmarking...')
    # #导出rulesets
    # haslib.dump_rulesets(rulesets,path=HS_RESOURCES)
    # #hyperscan test
    # cmd='./hs_test '+str(x)
    # os.system(cmd)

    print('---re benchmarking...')
    #TODO:compile pattern before test
    #TODO:设置阈值
    threshold=[0.0]*13 #暂初始化为0
    begin=datetime.datetime.now()
    haslib.test_python_re(rawstr_dataset,rulesets,threshold)
    total=datetime.datetime.now()-begin
    print('python re耗时:',total.seconds,'s',total.microseconds,'us')
print()



print('---bow模型report---')
# TODO:统计re匹配的各指标，与bow_rf_model比较,说明一致性。但模型正确率99%+，似乎没什么好对比的
y_pred=bow_rf_model.predict(X_test_bow) #ndarray
print(classification_report(y_test,y_pred))   # https://blog.csdn.net/weixin_43945848/article/details/122061718
print()


print('---完整方案report---')
#TODO:计算平均时延
topX_importances=haslib.get_topX_importances(classified_importances,topX=30)
rulesets=haslib.get_rulesets(topX_importances,bow_map,hash2word)
threshold=[0.0]*14
y_pred=[] #y_pred=np.empty((0,),dtype=np.int64)


for id,row in X_test_rawstr.iterrows():
    weighs=[0.0]* 14
    pred=-1 #初始预测分数
    for app,ruleset in rulesets.items():
        sum=0.0
        for pattern,weigh in ruleset.items():
            sum+= weigh if(re.search(pattern,row['raw_string'])) else 0
        weighs[app]=sum
        if weighs[app]>weighs[pred]: pred=app 
    pred=int(pred) 
    if(weighs[pred]>threshold[pred]):
        y_pred.append(pred)
    else:
        tmp=bow_rf_model.predict(X_test_bow.loc[[id]]) #未达到阈值时，再以模型预测
        y_pred+= tmp.tolist()
        
y_pred=np.array(y_pred)
print(classification_report(y_test,y_pred))













# %%
