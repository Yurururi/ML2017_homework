#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import string
import sys
import os
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Merge,Flatten
from keras.layers import GRU,LSTM
#from keras.layers.merge import Add
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam,SGD

from keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt 
import pickle
import itertools

train_path = sys.argv[1]
test_path = sys.argv[2]
output_path = sys.argv[3]
base_dir = os.path.dirname(os.path.realpath(__file__))
#==============================================================================
#    parameter
#==============================================================================
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 1000
batch_size = 128


#==============================================================================
#    Util
#==============================================================================
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

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

#==============================================================================
#    custom metrices
#==============================================================================
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.title(title)
    plt.colorbar()
    

#    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{}'.format(cm[i, j])[:4],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

#==============================================================================
#    Main
#==============================================================================
#if True:
def main():
    '''
    ### read training and testing data
    (Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    all_corpus = X_data + X_test
    print ('Find %d articles.' %(len(all_corpus)))
    
    ### tokenizer for all data
#    tokenizer = Tokenizer()
#    tokenizer.fit_on_texts(all_corpus)
#    word_index = tokenizer.word_index
    
    word_index = load_obj('word_index')
    num_words = len(word_index) + 1
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    ### convert word sequences to index sequence
    print ('Convert to index sequences.')
    train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    ### padding to equal length
    print ('Padding sequences.')
    train_sequences = pad_sequences(train_sequences)
    max_article_length = train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    
    ###
    train_tag = to_multi_categorical(Y_data,tag_list) 
    
    ### split data into training set and validation set
    (X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)
    
    ### get ebedding matrix from glove
    print ('Get embedding dict from glove.')
#    embedding_dict = get_embedding_dict('glove.6B.%dd.txt'%embedding_dim)
    embedding_dict = get_embedding_dict('glove.6B.%dd.txt'%embedding_dim)
    print ('Found %s word vectors.' % len(embedding_dict))
    num_words = len(word_index) + 1
    print ('Create embedding matrix.')
    embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)

    ### build model
    print ('Building model.')
    '''
    ### Load myparameters
    word_index = load_obj('word_index')
    num_words = len(word_index) + 1
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    tag_list = load_obj('tag_list')
    (_, X_test,_) = read_data(test_path,False)
    test_sequences = tokenizer.texts_to_sequences(X_test)
    test_sequences = pad_sequences(test_sequences,maxlen=306)
#==============================================================================
#    bidirectional
#==============================================================================
#    left = Sequential()
#    left.add(Embedding(num_words,
#                        embedding_dim,
#                        weights=[embedding_matrix],
#                        input_length=max_article_length,
#                        trainable=False))
#    left.add(GRU(128,activation='tanh',dropout=0.5))
#    right = Sequential()
#    right.add(Embedding(num_words,
#                        embedding_dim,
#                        weights=[embedding_matrix],
#                        input_length=max_article_length,
#                        trainable=False))
#    right.add(GRU(128,activation='tanh',dropout=0.5,go_backwards=True))
#    model = Sequential()
#    model.add(Merge([left,right], mode='sum'))
#    model.add(GRU(128,activation='tanh',dropout=0.5))
#    model.add(Dense(256,activation='relu'))
#    model.add(Dropout(0.1))
#    model.add(Dense(128,activation='relu'))
#    model.add(Dropout(0.1))
#    model.add(Dense(64,activation='relu'))
#    model.add(Dropout(0.1))
#    model.add(Dense(38,activation='sigmoid'))
#    model.summary()
    

#==============================================================================
#    RNN
#==============================================================================
    model = Sequential()
#    model.add(Embedding(num_words,
#                        embedding_dim,
#                        weights=[embedding_matrix],
#                        input_length=max_article_length,
#                        trainable=False))
    model.add(Embedding(num_words,
                        embedding_dim,
                        input_length=306,
                        trainable=False))
    model.add(GRU(128,return_sequences=True,activation='tanh',dropout=0.5))
    model.add(GRU(128,activation='tanh',dropout=0.5))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(38,activation='sigmoid'))
    model.load_weights('best_100dsig.hdf5')
#    model.summary()
#==============================================================================
#     Bag of words
#==============================================================================
#    model = Sequential()
#    model.add(Embedding(num_words,
#                        embedding_dim,
#                        weights=[embedding_matrix],
#                        input_length=max_article_length,
#                        trainable=False))
#    model.add(Flatten())
#    model.add(Dense(256,activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(256,activation='relu'))
#    model.add(Dropout(0.1))
#    model.add(Dense(256,activation='relu'))
#    model.add(Dropout(0.1))
#    model.add(Dense(128,activation='relu'))
#    model.add(Dropout(0.1))
#    model.add(Dense(64,activation='relu'))
#    model.add(Dropout(0.1))
#    model.add(Dense(38,activation='sigmoid'))
#    model.summary()
    
#==============================================================================
#    Train
#==============================================================================
#    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
#    model.compile(loss='categorical_crossentropy',
#                  optimizer=adam,
#                  metrics=[f1_score])
#    earlystopping = EarlyStopping(monitor='val_f1_score', patience = 25, verbose=1, mode='max')
#    checkpoint = ModelCheckpoint(filepath='best_100dsig.hdf5',
#                                 verbose=1,
#                                 save_best_only=True,
#                                 save_weights_only=True,
#                                 monitor='val_f1_score',
#                                 mode='max')
#    hist = model.fit(X_train, Y_train, 
#                     validation_data=(X_val, Y_val),
#                     epochs=nb_epoch, 
#                     batch_size=batch_size,
#                     callbacks=[earlystopping,checkpoint])
#    model.load_weights('best_100dsig.hdf5')

#    Y_train_pred = model.predict(X_train)

#    model = load_model('best_100dsig2L.h5')
    print('Predicting')
    Y_pred = model.predict(test_sequences)
    thresh = 0.4
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
#==============================================================================
#     Analyse tags
#==============================================================================
#    plt.figure(figsize=(12,10))
#    plt.bar(np.arange(0,38),np.sum(Y_train,axis=0))
#    cm = np.dot(Y_train.T,Y_train)
#    plt.figure(figsize=(26,22))
#    plot_confusion_matrix(cm, classes=tag_list,normalize=True,
#                      title='Covariance matrix')
    
if __name__=='__main__':
    main()