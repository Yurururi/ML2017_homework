# -*- coding: utf-8 -*-
"""
Created on Wed May 10 20:23:38 2017

@author: howard
"""

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.engine import Layer
from keras import regularizers
from keras import backend as K

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
        tr = 0.999
        alpha = 5
        return K.relu(inputs)+K.log(K.relu(inputs+1)+1-tr)\
            -K.relu(-inputs)*alpha
#        return K.relu(inputs, alpha=5)
#        return (inputs+K.log(inputs+1)+(inputs-K.log(inputs+1))*K.sign(inputs))/2
        
        
    def get_config(self):
        config = {'alpha': self.alpha}
        base_config = super(ielu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def local_model(k,h):
    model = Sequential()
    act = ielu()
    model.add(Dense(100,input_shape=(100,)))
    model.add(Activation(act))
    model.add(Dense(k))
    model.add(Activation(act))
    model.add(Dense(int(h),kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dense(k,kernel_regularizer=regularizers.l2(0.1)))
    model.add(Activation('elu'))
    model.add(Dense(100))
    model.add(Activation('elu'))
    model.add(Dense(100))
    model.compile(optimizer='adam', loss='mean_squared_logarithmic_error')
    model.summary()
    return model