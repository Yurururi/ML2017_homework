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

np.set_printoptions(threshold=200)
def grad_des(init_param,data,r_lambda=0):
    batch_size=150;tol=10**-8
    # Apply Adam
    turn = 0
    #theta_t = 0
    mv1 = 0 
    mv2 = 0
    
    grad_size = tol+1
    param = init_param
    best_grad_size = 100
    best_param = init_param
    alpha = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 10**-8
    #data_size = np.size(data[0],0)
    start = time.time()
    string = ''
    while grad_size >=tol and turn<10000:
        turn += 1
        #samp_idx = np.random.permutation(data_size)[:batch_size]
        #print('random_choice: ' + str(samp_idx))
        #batched_data = [data[k][samp_idx,:] for k in range(2)]
        #batched_data = data
        grad_vec = loss_f(param,data,1,r_lambda)
        mv1 = beta1*mv1 + (1-beta1)*grad_vec
        mv2 = beta2*mv2 + (1-beta1)*(grad_vec**2)
        mv1_h = mv1/(1-beta1**turn)
        mv2_h = mv2/(1-beta1**turn)
        param = param - alpha*mv1_h/(np.sqrt(mv2_h)+eps)
        grad_size = LA.norm(grad_vec)
        if grad_size < best_grad_size:
            best_grad_size = grad_size
            best_param = param
        #print(str(turn)+": " +'param: ' + str(param) +' gsize: ' + str(grad_size))
        #print('grad: ' + str(grad_vec))
        print("\b"*len(string),end = '')
        string = str(turn)+ ":" + str(grad_size)
        print(string, flush=True,end = '')
        #==== Adagrad =====
        #theta_t = np.sqrt(theta_t**2+grad_size**2)
        #lrn_rate = t_decay(turn)/(theta_t/np.sqrt(turn))
        #param = param - lrn_rate * grad_vec
        #==================
        
    end = time.time()
    print('\n' + str(turn) + ' turns')
    print('time: ' + str(end-start))
    print(best_grad_size)
    return best_param

def t_decay(t):
    etta = 10
    return etta/np.sqrt(t+1)
    #return etta
def loss_f(param,data,der=0,r_lambda = 0,ltype="cross-entropy"):
    if ltype == "mse":
        # MSE
        # l = |y-f(x)|^2, L' = -2(y-f(x))*f'(x) 
        x = data[0]; y = data[1]
        if der == 0:
            return LA.norm(y-lin_model_f(param,x))
        else:
            return np.array([np.sum(-2*(y-lin_model_f(param,x))*lin_model_f(param,x,i)) 
                             for i in range(len(param))])
    elif ltype == "cross-entropy":
        # Cross-entropy
        # l = C(f(x),y)
        #   = -[y*log(f(x))+(1-y)*(log(1-f(x)))], 
        # L' = -f'(x)*[(y-f(x))/(f(x)*(1-f(x)))] 
        # Add regularization term lambda * norm(param)
        x = data[0]; y = data[1]
        #r_lambda = 20
        if der == 0:
            return -np.dot(   y.T , np.log(  logist_model_f(param,x))) \
                   -np.dot( 1-y.T , np.log(1-logist_model_f(param,x))) \
                   +r_lambda * LA.norm(param)**2
        else:
            fx = logist_model_f(param,x)
            return np.array([np.sum(-(y.T-fx)*logist_model_f(param,x,i)) 
                             for i in range(len(param))]) + 2*r_lambda*param
def lin_model_f(param,data,der):
    b = param[0]; w = param[1:]
    if der < 0:
        return b + w.dot(data.T)
    elif der == 0:
        return np.ones(np.size(data,0))
    else:
        return data[:,der-1]
    
def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 0.00000000000001, 0.99999999999999)
def logist_model_f(param,data,der=-1):
    b = param[0]; w = param[1:]
    if der < 0:
        return sigmoid(b + w.dot(data.T))
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
    
    L = len(data_mean)
    return [np.reshape(data_mean,(L)), np.reshape(data_std,(L))]
def feature_regularize(data,ex_idx=[]):
    if type(data) is np.ndarray:
        L = np.size(data,1)
        '''
        return [data,[np.zeros(L),np.ones(L)]]
        '''
        fac = matrix_normalize_factor(data,ex_idx)
        return ((data-np.reshape(fac[0],(1,L)))/np.reshape(fac[1],(1,L)),fac)

def training_n_test(tr_X,tr_Y,tst_X,r_lambda=0):
    feature_nb = np.size(tr_X,1)
    sample_nb = np.size(tr_X,0)
    n_tr_X,fac = feature_regularize(tr_X)
    init_param = np.zeros(feature_nb + 1)
    param = grad_des(init_param,[n_tr_X,tr_Y],r_lambda)
    n_param = np.r_[param[0]-param[1:].dot(fac[0]/fac[1]),param[1:]/fac[1]]
    p_tr_Y = np.around(logist_model_f(n_param,tr_X))
    tr_acc = float((p_tr_Y==tr_Y.T).sum())/sample_nb
    print("Training set accuracy: ", tr_acc)
    tst_Y = logist_model_f(n_param,tst_X)
    return (tst_Y,n_param,tr_acc)

def dofeat(X):
    X = np.float_(X)
    #X[:,1] = np.log(X[:,1]+1)
    #X[:,3] = sigmoid(X[:,3])
    return X

def output_result(y,path):
    output_list = np.c_[(np.arange(len(y))+1).T,y.T].astype(int).tolist()
    output_df = pd.DataFrame(output_list, columns=["id","label"])
    output_df.to_csv(path,index=False)
    
if __name__ == "__main__":
    '''
    input_tr_data_file_dir = 'train.csv'
    input_tst_data_file_dir = 'test_X,csv'
    output_file_dir = 'testing_result,csv'
    df1 = pd.read_csv(input_tr_data_file_dir, delimiter = ',',encoding = 'cp950')
    raw_training_data = np.asarray(df1.values.tolist())
    dx_dic = {"age":0,"workclass":1,"fnlwgt":2,"education":3,"education_num":4, \
               "marital_status":5,"occupation":6,"relationship":7,"race":8, \
               "sex":9,"capital_gain":10,"capital_loss":11,"hours_per_week":12, \
               "native_country":13}
    print(np.unique(raw_training_data[:,3]))
    '''
    f1,f2,f3,f4 = sys.argv[1:]
    raw_tr_X = pd.read_csv(f1).values
    f_nb = np.size(raw_tr_X,1)
    tr_Y = pd.read_csv(f2, header=None).values
    raw_tst_X = pd.read_csv(f3).values;
    
    tr_X = dofeat(raw_tr_X)
    tst_X = dofeat(raw_tst_X)
    '''
    R_LAMBDA = [0.01,0.03,0.05,0.1,0.5,1,5,10,20,40,100]
    acc = np.zeros(len(R_LAMBDA))
    for k in range(len(R_LAMBDA)):
        tst_Y,param,acc[k] = training_n_test(tr_X,tr_Y,tst_X,R_LAMBDA[k])
    plt.plot(acc,'o-')
    '''
    #output_result(np.around(tst_Y),'Y_test1.csv')
    
    
    
    tst_Y,param,acc = training_n_test(tr_X,tr_Y,tst_X,20)
    output_result(np.around(tst_Y),f4)
    
    # cancel 18
    '''
    acc = np.zeros(f_nb); idx_ = [[]]*f_nb
    for k in [18]:
        print(k)
        idx = np.zeros(f_nb)==0
        idx[k] = False
        tr_X = raw_tr_X[:,idx]
        tst_X = raw_tst_X[:,idx]
        tst_Y,param,acc[k] = training_n_test(tr_X,tr_Y,tst_X)
        idx_[k] = idx
    best_acc = np.max(acc)
    best_idx = idx_[np.argmax(acc)]
    print("Best accuracy: ",np.max(acc))
    tst_Y,d,d = training_n_test(raw_tr_X[:,best_idx],tr_Y,raw_tst_X[:,best_idx])
    output_result(np.around(tst_Y),'Y_test.csv')
    '''