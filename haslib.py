import sys
import re
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer



def get_class_feature_importance(X,y,feature_importances):
    X=Normalizer(norm='l2').fit_transform(X) 
    classified_importances={}
    for app in set(y):
        class_importances=np.mean(X[y==app,:],axis=0)*feature_importances*10000
        classified_importances[app]=class_importances
    return classified_importances

def get_topX_importances(classified_importances,topX):
    topX_importances={}
    for app,class_importances in classified_importances.items():
        items=enumerate(class_importances.tolist())
        sorted_items=sorted(items,key=lambda x:x[1],reverse=True)
        if len(sorted_items)<topX:
            print("error: topX exceeds bow_id limit of app",app)
            sys.exit()
        topX_importances[app]=sorted_items[:topX]
    return topX_importances

def get_regular_expression(bow_id,bow_map,hash2word):
    words=hash2word[bow_map[bow_id]]
    if(type(words)is str):
        return re.escape(words.rstrip())
    else: 
        res=''
        for w in words:
            res=res+re.escape(w.rstrip())+'|'
        return res[:-1]

def get_rulesets(topX_importances,bow_map,hash2word):
    rulesets={}
    for app, id_weigh_list in topX_importances.items():
        ruleset={}
        for id,weigh in id_weigh_list:
            pattern=get_regular_expression(id,bow_map,hash2word)
            if(pattern.find(r' ') != -1): 
                #print(pattern)
                pattern=pattern.replace(r';\ charset=utf\-8', '')  #中间有空格导致不方便c++读入，又无重要意义，直接删除
                pattern=pattern.replace(r';\ charset=UTF\-8', '')
            ruleset[pattern] =weigh
        rulesets[app]=ruleset
    return rulesets

def dump_rulesets(rulesets,path):
    hs_rulesets=pd.DataFrame()
    for app,ruleset in rulesets.items():
        tmp=pd.DataFrame.from_dict(ruleset,orient='index',columns=['weigh'])
        tmp['pattern']=tmp.index
        tmp['label']=app
        hs_rulesets=pd.concat([hs_rulesets,tmp])
    hs_rulesets.sort_values(by='label')
    hs_rulesets.to_csv(path+'rulesets.txt',sep=' ',header=0,index=False)

def test_python_re(dataset,rulesets,threshold):
    true_shot=0
    true_miss=0
    false_shot=0
    false_miss=0
    num_apps=dataset.iloc[-1,-1]+1
    for _,row in dataset.iterrows():
        weighs=[0.0]* (num_apps+1)
        pred=-1
        for app,ruleset in rulesets.items():
            sum=0.0
            for pattern,weigh in ruleset.items():
                sum+= weigh if(re.search(pattern,row['raw_string'])) else 0
                # if sunday_algorithm(pattern,row['raw_string']):
                #     sum+=weigh
            weighs[app]=sum
            if weighs[app]>weighs[pred]: pred=app 

        pred=int(pred) #天坑...pred被转成了numpy int64 
        if pred is row['label']:
            if weighs[pred]> threshold[pred]: 
                true_shot+=1
            else: 
                true_miss+=1
        else:
            if weighs[pred]>threshold[pred]:
                false_shot+=1
                #print('false_shot',pred,row['raw_string'],row['label'])
            else: 
                false_miss+=1
                #print('false_miss',row['raw_string'],row['label'])
    print('true_shot,true_miss,false_shot,false_miss:',true_shot,true_miss,false_shot,false_miss)

def sunday_algorithm(pattern, text):
    m = len(pattern)
    n = len(text)
    # 初始化偏移数组，默认值为模式串长度+1
    shift = [m + 1] * 256

    # 构建坏字符规则表
    for i in range(m):
        shift[ord(pattern[i])] = m - i - 1

    # 匹配过程
    i = 0  # 文本串的索引
    while i <= n - m:
        j = 0  # 模式串的索引
        while j < m and text[i + j] == pattern[j]:
            j += 1
        if j == m:  # 匹配成功
            return True
        if j == 0:  # 没有字符匹配
            i += 1
        else:  # 根据坏字符规则移动
            i += max(1, shift[ord(text[i + j])])
    return False


def new_test_python_re(dataset,rulesets,threshold):
    true_shot=0
    true_miss=0
    false_shot=0
    false_miss=0
    num_apps=dataset.iloc[-1,-1]+1

    # 创建一个新字典来存储编译后的正则表达式对象
    compiled_patterns = {}

    # 遍历 rulesets 字典
    for app, patterns in rulesets.items():
        # 为每个应用程序创建一个字典来存储其编译后的正则表达式
        compiled_patterns[app] = {}
        # 遍历每个模式和对应的权重
        for pattern_str, weight in patterns.items():
            new_pattern = re.compile(pattern_str)
            # 编译正则表达式并存储在新字典中
            compiled_patterns[app][new_pattern] = weight

    begin = datetime.datetime.now()
    for _,row in dataset.iterrows():
        weighs=[0.0]* (num_apps+1)
        pred=-1
        for app,ruleset in compiled_patterns.items():
            sum=0.0
            for pattern,weigh in ruleset.items():
                sum+= weigh if(pattern.search(row['raw_string'])) else 0
            weighs[app]=sum
            if weighs[app]>weighs[pred]: pred=app

        pred=int(pred) #天坑...pred被转成了numpy int64
        if pred is row['label']:
            if weighs[pred]> threshold[pred]:
                true_shot+=1
            else:
                true_miss+=1
        else:
            if weighs[pred]>threshold[pred]:
                false_shot+=1
                #print('false_shot',pred,row['raw_string'],row['label'])
            else:
                false_miss+=1
                #print('false_miss',row['raw_string'],row['label'])
    total = datetime.datetime.now() - begin
    print('python re（纯匹配）耗时:', total.seconds, 's', total.microseconds, 'us')
    print('true_shot,true_miss,false_shot,false_miss:',true_shot,true_miss,false_shot,false_miss)