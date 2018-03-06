
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
        theta_t = np.sqrt(theta_t**2+grad_size**2)
        lrn_rate = t_decay(turn)/grad_size*inc
        old_grad_size = grad_size
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
        
        if old_grad_size <= grad_size:
            inc = inc * 0.5
        else:
            inc = inc * 1.7
        turn = turn + 1
    end = time.time()
    '''
    print(str(turn) + ' turns')
    print('time: ' + str(end-start))
    print(grad_size)
    '''
    return param
def t_decay(t):
    etta = 1
    return etta/np.sqrt(np.sqrt(t+1))
    #return etta
    

def loss_f(param,data,der=0):
    # l = |y-f(x)|^2, L' = -2(y-f(x))*f'(x) 
    x = data[0]; y = data[1]
    if der == 0:
        return LA.norm(y-model_f(param,x))
    else:
        return np.array([np.sum(-2*(y-model_f(param,x))*model_f(param,x,i)) 
                         for i in range(len(param))])
    
class model_data_dic:
    lin_mdl_data = np.ndarray(0)
    sine_mdl_data = np.ndarray(0)
    
def model_f(param,data,der=-1):
    if type(data) is np.ndarray:
        return lin_model_f(param,data,der)
    elif type(data) is model_data_dic:
        result = 0
        if data.lin_mdl_data.size != 0:
            L = np.size(data.lin_mdl_data,1)+1
            if der ==-1:
                result = result + lin_model_f(param[:L],data.lin_mdl_data,der)
            elif der < L:
                result = result + lin_model_f(param[:L],data.lin_mdl_data,der)
                der = der-L-1
            elif der >= L:
                der = der-L
                param = param[L:]
        if data.sine_mdl_data.size != 0:
            L = int(np.size(data.sine_mdl_data,1)/2)+1
            if der ==-1:
                result = result + sine_model_f(param[:L],data.sine_mdl_data,der)
            elif der < L: 
                result = result + sine_model_f(param[:L],data.sine_mdl_data,der)
                der = der-L-1
            elif der >= L:
                der = der-L
                param = param[L:]
        return result
def lin_model_f(param,data,der):
    b = param[0]; w = param[1:]
    if der < 0:
        return b + w.dot(data.T)
    elif der == 0:
        return np.ones(np.size(data,0))
    else:
        return data[:,der-1]
    
def sine_model_f(param,data,der):
    # f =  amplitude_size * wind_speed * sin(wind_dir + phase)
    # df/d(phase) =  amplitude_size * wind_speed * cos(wind_dir + phase)
    
    phs = param[0]; amp = param[1:]
    half_idx = int(np.size(data,1)/2);
    ws = data[:,:half_idx]; wd = data[:,half_idx:]*np.pi/180
    
    if der < 0:
        return np.sum(amp * ws * np.sin(wd + 2*np.pi* phs),1)
    elif der == 0:
        return 2*np.pi*np.sum(amp * ws * np.cos(wd + 2*np.pi* phs),1)
    else:
        return ws[:,der-1]*np.sin(wd[:,der-1] + 2*np.pi* phs)
    
    
def poly_model_f(param,data):
    print('')
    
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
    elif type(data) is model_data_dic:
        result_data = model_data_dic()
        fac = model_data_dic()
        if data.lin_mdl_data.size != 0:
            tmp = feature_normalize(data.lin_mdl_data)
            result_data.lin_mdl_data = tmp[0]
            fac.lin_mdl_data = tmp[1]
        if data.sine_mdl_data.size != 0:
            ex_idx = range(int(np.size(data.sine_mdl_data,1)/2),np.size(data.sine_mdl_data,1))
            tmp = feature_normalize(data.sine_mdl_data,ex_idx)
            result_data.sine_mdl_data = tmp[0]
            fac.sine_mdl_data = tmp[1]
        return [result_data,fac]
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
    elif type(s_data) is list:
        self_s_tr_X = model_data_dic()
        self_s_tst_data = model_data_dic()
        if s_data[0].lin_mdl_data.size != 0:
            self_s_tr_X.lin_mdl_data = s_data[0].lin_mdl_data[tr_idx,:]
            self_s_tst_data.lin_mdl_data = s_data[0].lin_mdl_data[tst_idx,:]
        if s_data[0].sine_mdl_data.size != 0:
            self_s_tr_X.sine_mdl_data = s_data[0].sine_mdl_data[tr_idx,:]
            self_s_tst_data.sine_mdl_data = s_data[0].sine_mdl_data[tst_idx,:]
        self_s_tr_Y = s_data[1][tr_idx]
        self_s_tr_data = [self_s_tr_X,self_s_tr_Y,s_data[2]]
        ground_true_data = s_data[1][tst_idx]
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
    elif type(s_tr_data) is list:
        s_data_fac = feature_normalize(s_tr_data[0])
        init_param = np.random.random(s_tr_data[2])
        tr_data = [s_data_fac[0],s_tr_data[1]]
    else:
        print('DATA TYPE IS WRONG!!!')
        return np.zeros(1,10)
    
    if type(s_tst_data) is np.ndarray:
        L = len(s_data_fac[1][0])
        n_tst_data = (s_tst_data-np.reshape(s_data_fac[1][0],(1,L))) \
                      /np.reshape(s_data_fac[1][1],(1,L))
    elif type(s_tst_data) is model_data_dic:
        n_tst_data = model_data_dic()
        
        if s_tst_data.lin_mdl_data.size != 0:
            tmp_fac = s_data_fac[1].lin_mdl_data
            L = len(tmp_fac[0])
            n_tst_data.lin_mdl_data = (s_tst_data.lin_mdl_data-np.reshape(tmp_fac[0],(1,L))) \
                      /np.reshape(tmp_fac[1],(1,L))
        if s_tst_data.sine_mdl_data.size != 0:
            tmp_fac = s_data_fac[1].sine_mdl_data
            L = len(tmp_fac[0])
            n_tst_data.sine_mdl_data = (s_tst_data.sine_mdl_data-np.reshape(tmp_fac[0],(1,L))) \
                      /np.reshape(tmp_fac[1],(1,L))
        
    else:
        print('DATA TYPE IS WRONG!!!')
        return np.ones(1,20)
    
    #result_param = grad_des(init_param,tr_data)
    #print(result_param)
    result_param = LA.lstsq(np.c_[np.ones((len(tr_data[1]),1)),tr_data[0].lin_mdl_data],tr_data[1])[0]
    
    predicted_result = model_f(result_param,n_tst_data)
    
    return predicted_result
    
def take_all_data(data):
    data_size = np.size(data)
    s_data = np.ndarray((0,9))
    for k in range(data_size-9):
        s_data = np.r_[s_data,data[:,k:(9+k)]]
    return s_data
if __name__ == "__main__":
    #input_argv_list = sys.argv
    #input_tr_data_file_dir = input_argv_list[1]
    #input_tst_data_file_dir = input_argv_list[2]
    #output_file_dir = input_argv_list[3]
    
    input_tr_data_file_dir , input_tst_data_file_dir , output_file_dir = sys.argv[1:]
    #input_tr_data_file_dir = 'train.csv'
    #input_tst_data_file_dir = 'test_X,csv'
    #output_file_dir = 'testing_result,csv'
    
    df1 = pd.read_csv(input_tr_data_file_dir, delimiter = ',',encoding = 'cp950')
    training_data = np.asarray(df1.values.tolist())
    pm25_data = training_data[training_data[:,2]=='PM2.5',3:].astype(np.float)
    data_size = np.size(pm25_data)
    pm25_data = pm25_data.reshape((1,data_size))
    pm10_data = training_data[training_data[:,2]=='PM10',3:].astype(np.float).reshape((1,data_size))
    CO_data = training_data[training_data[:,2]=='CO',3:].astype(np.float).reshape((1,data_size))
    NO_data = training_data[training_data[:,2]=='NO',3:].astype(np.float).reshape((1,data_size))
    NO2_data = training_data[training_data[:,2]=='NO2',3:].astype(np.float).reshape((1,data_size))
    NOx_data = training_data[training_data[:,2]=='NOx',3:].astype(np.float).reshape((1,data_size))
    w_dir_data = training_data[training_data[:,2]=='WIND_DIREC',3:].astype(np.float).reshape((1,data_size))
    w_sp_data = training_data[training_data[:,2]=='WIND_SPEED',3:].astype(np.float).reshape((1,data_size))
    #'''
    phs = 0.6*np.pi
    w_data = w_sp_data*np.sin(w_dir_data*np.pi/180 + phs)
    s_w_data = take_all_data(w_data)
    s_pm25_data = take_all_data(pm25_data)
    s_pm10_data = take_all_data(pm10_data)
    s_CO_data = take_all_data(CO_data)
    s_NO_data = take_all_data(NO_data)
    s_NO2_data = take_all_data(NO2_data)
    s_NOx_data = take_all_data(NOx_data)
    s_tr_data_t = np.c_[s_pm25_data,s_pm10_data,s_w_data]
    ''' 
    s_w_data = np.c_[w_sp_data[:,:9],w_dir_data[:,:9]]
    s_pm25_data = pm25_data[:,:10]
    for k in range(1,15):
        s_pm25_data = np.r_[s_pm25_data,
                            pm25_data[:,k:(10+k)]]
        s_w_data = np.r_[s_w_data,
                      np.c_[w_sp_data[:,k:(9+k)],w_dir_data[:,k:(9+k)]]]
    phs = 0.8*np.pi
    s_w_data = s_w_data[:,:9]*np.sin(s_w_data[:,9:]*np.pi/180 + phs)
    s_tr_data = np.c_[s_pm25_data[:,:-1],s_w_data]
    '''
    NN = 8
    
    tr_X = [model_data_dic()]*NN
    tr_X[0].lin_mdl_data = np.c_[s_pm25_data]
    tr_X[1].lin_mdl_data = np.c_[s_pm25_data,s_w_data]
    tr_X[2].lin_mdl_data = np.c_[s_pm25_data,s_pm10_data]
    tr_X[3].lin_mdl_data = np.c_[s_pm25_data,s_CO_data]
    tr_X[4].lin_mdl_data = np.c_[s_pm25_data,s_NO_data]
    tr_X[5].lin_mdl_data = np.c_[s_pm25_data,s_NO2_data]
    tr_X[6].lin_mdl_data = np.c_[s_pm25_data,s_NOx_data]
    tr_X[7].lin_mdl_data = np.c_[s_pm25_data,s_pm10_data,s_CO_data,s_NO_data,s_NO2_data,s_NOx_data]
    
    #tr_X.sine_mdl_data = s_w_data
    s_tr_data = [[]]*NN
    tr_Y = pm25_data[:,9:].flatten()
    for k in range(NN):
        s_tr_data[k] = [tr_X[k],tr_Y,60]
    
    
    df2 = pd.read_csv(input_tst_data_file_dir, delimiter = ',',encoding = 'cp950')
    testing_data = np.asarray(df2.values.tolist())
    tst_pm25_data = testing_data[testing_data[:,1]=='PM2.5',2:].astype(np.float)
    tst_pm25_data = testing_data[testing_data[:,1]=='PM2.5',2:].astype(np.float)
    tst_ws_data = testing_data[testing_data[:,1]=='WIND_SPEED',2:].astype(np.float)
    tst_wd_data = testing_data[testing_data[:,1]=='WIND_DIREC',2:].astype(np.float)
    tst_w_data = tst_ws_data*np.sin(tst_wd_data*np.pi/180 + phs)
    s_tst_data = model_data_dic()
    s_tst_data.lin_mdl_data = np.c_[tst_pm25_data,tst_w_data]
    #s_tst_data.sine_mdl_data = tst_w_data
    
    NN = 8;
    tr_size = np.arange(0.2,0.9,0.1);
    err = np.zeros((int(len(tr_size)),NN));
    labels = ['Only PM2.5','With wind data','With PM10','With CO','With NO','With NO2','With NOx','Mix All']
    hds = [[]]*NN
    for kk in range(NN):
        '''
        for k in range(len(tr_size)):
            err[k,kk] = self_training_test(s_tr_data[kk],[0.1,tr_size[k]])
        '''
        hds[kk], = plt.plot(tr_size,err[:,kk],'^-',label = labels[kk])
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    fig.savefig('test2png.png', dpi=100)
    plt.xlabel('training size')
    plt.ylabel('RSME')
    plt.legend(handles = hds)
    plt.show()
    '''
    predicted_pm25 = training_n_reconstruct(s_tr_data,s_tst_data)
    
    id_s = testing_data[testing_data[:,1]=='PM2.5',0]
    output_list = np.c_[id_s,predicted_pm25].tolist()
    output_df = pd.DataFrame(output_list, columns=["id","value"])
    output_df.to_csv(output_file_dir,index=False)
    '''