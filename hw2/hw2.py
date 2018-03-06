# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 21:27:42 2017

@author: howard
"""
import numpy as np
from numpy import linalg as LA
import pandas as pd
import time
import sys
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.inf)
def grad_des(init_param,data,tol=10**-8):
    param = init_param
    f_value = loss_f(param,data)
    grad_vec = loss_f(param,data,1)
    grad_size = LA.norm(grad_vec)
    turn = 0; theta_t = 0; inc = 1
    start = time.time()
    while grad_size >=tol and turn<= 100000:
        #print(param)
        #print(str(f_value)+' ' + str(grad_size)+' '+str(theta_t)+' '+str(inc))
        #if np.mod(turn,1000) == 0:
        #    theta_t = 0
        turn += turn 
        theta_t = np.sqrt(theta_t**2+grad_size**2)
        lrn_rate = t_decay(turn)/grad_size/np.sqrt(turn)
        new_param = param - lrn_rate * grad_vec
        new_f_value = loss_f(new_param,data)
        if f_value < new_f_value :
            inc = inc
        else:
            inc = inc
        param = new_param
        f_value = new_f_value
        grad_vec = loss_f(param,data,1)
        grad_size = LA.norm(grad_vec)
    end = time.time()
    #'''
    print(str(turn) + ' turns')
    print('time: ' + str(end-start))
    print(grad_size)
    #'''
    return param
def t_decay(t):
    etta = 1
    return etta/np.sqrt(np.sqrt(t+1))
    #return etta
def loss_f(param,data,der=0,ltype="mse"):
    if ltype == "mse":
        # MSE
        # l = |y-f(x)|^2, L' = -2(y-f(x))*f'(x) 
        x = data[0]; y = data[1]
        if der == 0:
            return LA.norm(y-model_f(param,x))
        else:
            return np.array([np.sum(-2*(y-lin_model_f(param,x))*lin_model_f(param,x,i)) 
                             for i in range(len(param))])
    elif ltype == "cross-entropy":
        # Cross-entropy
        # l = C(f(x),y)
        #   = -[y*log(f(x))+(1-y)*(log(1-f(x)))], 
        # L' = -f'(x)*[(y-f(x))*(f(x)*(1-f(x)))] 
        x = data[0]; y = data[1]
        if der == 0:
            return -np.dot(   y , np.log(  logist_model_f(param,x))) \
                   -np.dot( 1-y , np.log(1-logist_model_f(param,x)))
        else:
            return np.array([np.sum(-2*(y-lin_model_f(param,x))*lin_model_f(param,x,i)) 
                             for i in range(len(param))])
    
        
def model_f(param,data,der=-1):
    if type(data) is np.ndarray:
        return lin_model_f(param,data,der)
def lin_model_f(param,data,der):
    b = param[0]; w = param[1:]
    if der < 0:
        return b + w.dot(data.T)
    elif der == 0:
        return np.ones(np.size(data,0))
    else:
        return data[:,der-1]
def logist_model_f(param,data,der):
    b = param[0]; w = param[1:]
    if der < 0:
        return 1-(1 + np.exp(b + w.dot(data.T)))
    elif der == 0:
        return np.ones(np.size(data,0))
    else:
        return data[:,der-1]
    
    
    
def matrix_normalize_factor(data,ex_idx):
    data_mean = np.mean(data,axis=0)
    data_std = np.std(data,axis=0)
    data_std[data_std==0] = 1
    
    data_std[ex_idx] = 1
    data_mean[ex_idx] = 0
    
    #print(np.min(data_std))
    #print(data_mean)
    
    L = len(data_mean)
    return [np.reshape(data_mean,(L)), np.reshape(data_std,(L))]
    
def feature_normalize(data,ex_idx=[]):
    if type(data) is np.ndarray:
        fac = matrix_normalize_factor(data,ex_idx)
        L = len(fac[0])
        return [(data-np.reshape(fac[0],(1,L)))/np.reshape(fac[1],(1,L)),fac]
def training_test(data_set):
    tst_result = training_n_reconstruct(data_set[0],data_set[1])
    tst_err = LA.norm(tst_result - data_set[2])/np.sqrt(len(data_set[2]))
    return tst_err

def split_data(s_data,tst_idx,tr_idx = None):
    if tr_idx is None:
        tr_idx = np.logical_not(tst_idx)
    if type(s_data) is np.ndarray: 
        self_s_tr_data = s_data[tr_idx,:]
        self_s_tst_data = s_data[tst_idx,:-1]
        ground_true_data = s_data[tst_idx:,-1]
    return [self_s_tr_data,self_s_tst_data,ground_true_data]
def self_training_test(s_data,split_size=[0.1,0.9]):
    if type(s_data) is np.ndarray: 
        L = np.size(s_data,0)
    elif type(s_data) is list:
        L = np.size(s_data[1],0)
    tst_size = split_size[0]; tr_size = split_size[1]
    print('Self training testing...,' \
          + 'training size=' + str(tr_size)
          + ',test size=' + str(tst_size))
    self_tst_err = np.ndarray((1,0))
    for k in range(500):
        tmp =  np.random.permutation(L)
        tst_idx = np.zeros(L,dtype=bool)
        tst_idx[tmp[:int(L*tst_size)]] = True
        tr_idx = np.zeros(L,dtype=bool)
        tr_idx[tmp[int(L*tst_size):int(L*(tr_size+tst_size))]] = True 
        #print('size = ' + str(tmp.size))
        data_set = split_data(s_data,tst_idx,tr_idx)
        self_tst_err = np.c_[self_tst_err,training_test(data_set)]
    self_tst_err = np.mean(self_tst_err)
    
    print('Self testing error : ' + str(self_tst_err))
    
    return self_tst_err
def training_n_reconstruct(s_tr_data,s_tst_data):
    if type(s_tr_data) is np.ndarray:
        tr_Y = s_tr_data[:,-1]
        s_data_fac = feature_normalize(s_tr_data[:,:-1])
        tr_X = s_data_fac[0]
        
        tr_data = [tr_X,tr_Y]
        init_param = np.random.random(np.size(tr_X,axis = 1)+1)
    else:
        print('DATA TYPE IS WRONG!!!')
        return np.zeros(1,10)
    
    if type(s_tst_data) is np.ndarray:
        L = len(s_data_fac[1][0])
        n_tst_data = (s_tst_data-np.reshape(s_data_fac[1][0],(1,L))) \
                      /np.reshape(s_data_fac[1][1],(1,L))
        
    else:
        print('DATA TYPE IS WRONG!!!')
        return np.ones(1,20)
    
    result_param = grad_des(init_param,tr_data)
    print(result_param)
    #result_param = LA.lstsq(np.c_[np.ones((len(tr_data[1]),1)),tr_data[0].lin_mdl_data],tr_data[1])[0]
    
    predicted_result = model_f(result_param,n_tst_data)
    
    return predicted_result
    
def take_all_data(data):
    data_size = np.size(data)
    s_data = np.ndarray((0,9))
    for k in range(data_size-9):
        s_data = np.r_[s_data,data[:,k:(9+k)]]
    return s_data
if __name__ == "__main__":
    input_tr_data_file_dir = 'train.csv'
    input_tst_data_file_dir = 'test_X,csv'
    output_file_dir = 'testing_result,csv'
    df1 = pd.read_csv(input_tr_data_file_dir, delimiter = ',',encoding = 'cp950')
    raw_training_data = np.asarray(df1.values.tolist())
    idx_dic = {"age":0,"workclass":1,"fnlwgt":2,"education":3,"education_num":4, \
               "marital_status":5,"occupation":6,"relationship":7,"race":8, \
               "sex":9,"capital_gain":10,"capital_loss":11,"hours_per_week":12, \
               "native_country":13}
    print(np.unique(raw_training_data[:,3]))