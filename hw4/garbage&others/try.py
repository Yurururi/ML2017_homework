# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:58:19 2017

@author: howard
"""

from keras.engine import Layer
from keras import backend as K
import time

class ielu(Layer):
    """IELU version of a Rectified Linear Unit.
    It allows a small gradient when the unit is not active:
    `f(x) = alpha * x for x < 0`,
    `f(x) = x for x >= 0`.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        alpha: float >= 0. Negative slope coefficient.
    # References
        - [Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf)
    """

    def __init__(self, **kwargs):
        super(ielu, self).__init__(**kwargs)
        self.supports_masking = True
            
    def call(self, inputs):
        tr = 0.9
        alpha = 1
        return inputs*K.relu(inputs)+K.log(K.relu(inputs+1)+1-tr)-K.relu(-inputs)*alpha
#        return K.relu(inputs, alpha=5)
#        return (inputs+K.log(inputs+1)+(inputs-K.log(inputs+1))*K.sign(inputs))/2
        
        
    def get_config(self):
        config = {'alpha': self.alpha}
        base_config = super(ielu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



from keras.models import Sequential
from keras.layers import Dense,Activation
from keras import regularizers
#import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import mymodel

data = np.load('data.npz')


list_name = 'loss_list'
try:
    c_loss_list = np.load(list_name+'.npy').tolist()
    n1 = 0
    while np.size(c_loss_list[n1])>0:
        n1+=1
except:
    c_loss_list = [[]]*200
    n1 = 0
print('Start at {}'.format(n1))
for i in np.arange(n1,200):
    stt = time.time()
    x = data[str(i)]
    k = 79
    ubd = 60
    lbd = 0
    c_losses = np.array([])
    c_c_size = np.array([])
    ssl = [-30,-15,-8,-4,-2,-1]
    for itr,step_size in enumerate(ssl):
        losses = np.array([])
        c_size = np.arange(ubd,lbd+step_size,step_size)
        del_count = 0
        for ih,h in enumerate(c_size):
            if h in c_c_size:
                c_size = np.delete(c_size,ih-del_count)
                del_count+=1
                continue
            if h==0:
                h=1
                c_size[ih]=1
                
            model1 = mymodel.local_model(k,h)
            
            
            his = model1.fit(x,x,epochs=15,batch_size=int(np.size(x,0)/20))
            losses = np.append(losses,his.history['loss'][-1])
            del model1,his
        c_losses = np.append(c_losses,losses)
        c_c_size = np.append(c_c_size,c_size)
        idx = np.argsort(c_c_size)[::-1]
        c_losses = c_losses[idx]
        c_c_size = c_c_size[idx]
        del losses,c_size,idx
        if step_size!=-1:
            gap = (c_losses[2:]-2*c_losses[1:-1]+c_losses[:-2])/(c_c_size[:-2]-c_c_size[2:])
            midx = np.argmax(gap)
            ubd = c_c_size[midx+1]-ssl[itr+1]
            lbd = c_c_size[midx+1]+ssl[itr+1]
    c_loss_list[i]=[c_c_size,c_losses]
    np.save(list_name,c_loss_list)
#    ttgap = (c_losses[1:]-c_losses[:-1])/(c_c_size[:-1]-c_c_size[1:])
#    ttmidx = np.argmax(ttgap)
#    output_dim = c_c_size[ttmidx]
#    print('DIM = {} '.format(output_dim))
#    plt.plot(c_c_size,c_losses,'ro')
    
