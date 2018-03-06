# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 13:07:10 2017

@author: howard
"""
import numpy as np
from numpy import linalg as LA
import pandas as pd
import time
import sys
import matplotlib.pyplot as plt

idx_dic = {"age":0,"workclass":1,"fnlwgt":2,"education":3,"education_num":4, \
           "marital_status":5,"occupation":6,"relationship":7,"race":8, \
           "sex":9,"capital_gain":10,"capital_loss":11,"hours_per_week":12, \
           "native_country":13}

np.set_printoptions(threshold=1000)


def load_data(args):
    tr_X = pd.read_csv(args[0]).values
    tr_Y = pd.read_csv(args[1],header=None).values
    tst_X = pd.read_csv(args[2]).values
    return (tr_X, tr_Y, tst_X)

def get_model_param(tr_X,tr_Y):
    # Gaussian distribution parameters
    c1_idx = (tr_Y==1).flatten()
    c2_idx = (tr_Y==0).flatten()
    N1 = float(np.count_nonzero(c1_idx)); N2 = float(np.count_nonzero(c2_idx))
    miu1 = np.mean(tr_X[c1_idx,:],0)
    miu2 = np.mean(tr_X[c2_idx,:],0)
    #print(np.transpose(tr_X[c1_idx,:]-miu1).shape)
    #print((tr_X[c1_idx,:]-miu1).shape)
    #sigma1 = np.dot(np.transpose(tr_X[c1_idx,:]-miu1),tr_X[c1_idx,:]-miu1)/N1
    #sigma2 = np.dot(np.transpose(tr_X[c2_idx,:]-miu2),tr_X[c2_idx,:]-miu2)/N2
    sigma1 = 0; sigma2 = 0
    for i in range(int(N1+N2)):
        if tr_Y[i] == 1:
            sigma1 += np.dot(np.transpose([tr_X[i] - miu1]), [(tr_X[i] - miu1)])
        else:
            sigma2 += np.dot(np.transpose([tr_X[i] - miu2]), [(tr_X[i] - miu2)])
    sigma1 /= N1; sigma2 /= N2; 
    shared_sigma = N1/(N1+N2)*sigma1 + (N2/(N1+N2))*sigma2
    #miu1 /= sc; miu2 /= sc; shared_sigma /= sc**2
    return [miu1,miu2,shared_sigma,N1,N2]

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)

def predict_result(tst_X,param):
    miu1,miu2,shared_sigma,N1,N2 = param
    sigma_inv = np.linalg.inv(shared_sigma)
    w = np.dot( (miu1-miu2), sigma_inv)
    b = -0.5 * np.dot(np.dot(miu1, sigma_inv), miu1) + \
         0.5 * np.dot(np.dot(miu2, sigma_inv), miu2) + np.log(N1/N2)
    
    return sigmoid(np.dot(w, tst_X.T) + b)

def output_result(y,path):
    output_list = np.c_[(np.arange(len(y))+1).T,y.T].astype(int).tolist()
    output_df = pd.DataFrame(output_list, columns=["id","label"])
    output_df.to_csv(path,index=False)

if __name__ == "__main__":
    '''
    input_tr_data_file_dir = 'train.csv'
    input_tst_data_file_dir = 'test_X,csv'
    output_file_dir = 'testing_result,csv'
    df1 = pd.read_csv(f1, delimiter = ',',encoding = 'cp950')
    raw_training_data = np.asarray(df1.values.tolist())
    '''
    args = sys.argv[1:]
    #df = pd.read_csv('X_train', delimiter = ',',encoding = 'cp950')
    tr_X,tr_Y,tst_X = load_data(args[:3])
    #print(np.unique(raw_training_data[:,3]))
    
    param = get_model_param(tr_X,tr_Y)
    #y = predict_result(tr_X,param)
    #result = tr_Y.flatten() == np.around(y)
    #print('Training accuracy: ',float(result.sum())/len(result))
    tst_Y = predict_result(tst_X,param)
    output_result(np.around(tst_Y),args[3])
    #y = predict_result(tst_X,param)
    
    