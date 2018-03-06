# -*- coding: utf-8 -*-
"""
Created on Sat May 20 23:28:51 2017

@author: howard
"""

import word2vec

train_path = 'train_data.csv'
test_path = 'test_data.csv'


def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r',encoding='utf-8') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

if True:
    (Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    