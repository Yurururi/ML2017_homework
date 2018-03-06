import numpy as np
from numpy import linalg as LA
import pandas as pd
import time
np.set_printoptions(threshold=np.inf)
def grad_des(init_param,data,tol=10**-8):
    param = init_param
    grad_vec = loss_f(param,data,1)
    grad_size = LA.norm(grad_vec)
    turn = 0; theta_t = 0; inc = 1
    start = time.time()
    while grad_size >=tol and turn<= 1000000:
        #print(param)
        print(str(grad_size)+' '+str(inc))
        theta_t = np.sqrt(theta_t**2+grad_size**2)
        lrn_rate = t_decay(turn)/grad_size*inc
        param = param - lrn_rate * grad_vec
        grad_vec = loss_f(param,data,1)
        old_grad_size = grad_size
        grad_size = LA.norm(grad_vec)
        if old_grad_size < grad_size:
            inc = inc / 2
        else:
            inc = inc * 1.7
        turn = turn + 1
    end = time.time()
    print(str(turn) + ' turns')
    print('time: ' + str(end-start))
    print(grad_size)
    return param
def t_decay(t):
    etta = 1
    return etta/np.sqrt(np.sqrt(t+1))

def loss_f(param,data,der=0):
    # l = |y-f(x)|^2, L' = -2(y-f(x))*f'(x) 
    x = data[0]; y = data[1]
    if der == 0:
        return LA.norm(y-model_f(param,x))
    else:
        return np.array([np.sum(-2*(y-model_f(param,x))*model_f(param,x,i)) 
                         for i in range(len(param))])
    
def model_f(param,data,der=-1):
    return lin_model_f(param,data,der)
    
def lin_model_f(param,data,der):
    b = param[0]; w = param[1:]
    if der < 0:
        return b + w.dot(data.T)
    elif der == 0:
        return np.ones(np.size(data,0))
    else:
        return data[:,der-1]
    
def sine_model_f(param,data,der):
    # f =  weight_size * wind_speed*
    amp = param[::2]; phs = param[1::2]
    
    if der < 0:
        return 
    elif der == 0:
        return np.ones(np.size(data,0))
    else:
        return data[:,der-1]
    
    
def poly_model_f(param,data):
    print('')
    
def matrix_normalize_factor(data):
    data_mean = np.mean(data,axis=0)
    data_std = np.std(data,axis=0)
    data_std[data_std==0] = 1
    #print(np.min(data_std))
    #print(data_mean)
    
    L = len(data_mean)
    return [np.reshape(data_mean,(L)), np.reshape(data_std,(L))]
    
def feature_normalize(data,end_line=0):
    if end_line == 0:
        fac = matrix_normalize_factor(data)
    else:
        fac = matrix_normalize_factor(data[:,:end_line])
    L = len(fac[0])
    return [ (data-np.reshape(fac[0],(1,L)))/np.reshape(fac[1],(1,L)), 
              [fac[0], fac[1]]]
    
def training_n_reconstruct(s_tr_data,s_tst_data):
    s_data_fac = feature_normalize(s_tr_data)
    n_s_data = s_data_fac[0]
    tr_X = n_s_data[:,:-1]
    tr_Y = n_s_data[:,-1]
    tr_data = [tr_X,tr_Y]
    
    init_param = np.random.random(np.size(tr_X,axis = 1)+1)
    result_param = grad_des(init_param,tr_data)
    
    L = len(s_data_fac[1][0])-1
    n_tst_data = (s_tst_data-np.reshape(s_data_fac[1][0][:-1],(1,L))) \
                 /np.reshape(s_data_fac[1][1][:-1],(1,L))
    
    predicted_result = model_f(result_param,n_tst_data)
    predicted_result = predicted_result*s_data_fac[1][1][-1]+s_data_fac[1][0][-1]
    
    return predicted_result
    
if __name__ == "__main__":
    df = pd.read_csv('train.csv', delimiter = ',',encoding = 'cp950')
    training_data = np.asarray(df.values.tolist())
    pm25_data = training_data[training_data[:,2]=='PM2.5',3:].astype(np.float)
    pm10_data = training_data[training_data[:,2]=='PM10',3:].astype(np.float)
    s_pm25_data = pm25_data[:,:10]
    for k in range(1,15):
        s_pm25_data = np.r_[s_pm25_data,
                            np.c_[pm25_data[:,k:(10+k)]]]
    s_tr_data = np.c_[s_pm25_data]
    
    df = pd.read_csv('test_X.csv', delimiter = ',',encoding = 'cp950')
    testing_data = np.asarray(df.values.tolist())
    tst_pm25_data = testing_data[testing_data[:,1]=='PM2.5',2:].astype(np.float)
    s_tst_data = np.c_[tst_pm25_data]
    
    predicted_pm25 = training_n_reconstruct(s_tr_data,s_tst_data)
    
    id_s = testing_data[testing_data[:,1]=='PM2.5',0]
    output_list = np.c_[id_s,predicted_pm25].tolist()
    output_df = pd.DataFrame(output_list, columns=["id","value"])
    output_df.to_csv('testing_result.csv',index=False)
    