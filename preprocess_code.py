import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pandas as pd
import time
import matplotlib.pyplot as plt


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary
    
    train = []
    test = []
    train_data = np.zeros((50000,np.shape(mat["train0"])[1]))
    validation_data = np.zeros((10000,np.shape(mat["train0"])[1]))
    test_data = np.zeros((10000,np.shape(mat["test0"])[1]))
    train_label = np.zeros((50000,))
    validation_label = np.zeros((10000,))
    test_label = np.zeros((10000,))
    train_len=0
    validation_len=0
    test_len=0
    train_len_l=0
    validation_len_l=0
    print(np.shape(train_label))
    for key in mat:
        if(key[-2]=="n"):
            train = np.zeros(np.shape(mat.get(key)))
            shuffle = np.random.permutation(range(train.shape[0]))
            train_data[train_len:train_len+len(train)-1000] = train[shuffle[0:len(train)-1000],:]
            train_label[train_len_l:train_len_l+len(train)] = key[-1]
            print(np.shape(train_label))
            validation_data[validation_len:validation_len+1000]=train[shuffle[len(train)-1000:len(train)],:]
            validation_len=validation_len+1000
            validation_label[validation_len_l:validation_len_l+1000] = key[-1]
            validation_len_l=validation_len_l+1000
            train_len=train_len+len(train)-1000
            train_len_l=train_len_l+len(train)-1000
        elif(key[-2]=="t"):
            test = np.zeros(np.shape(mat.get(key)))
            test_data[test_len:test_len+len(test)] = test[np.random.permutation(range(test.shape[0])),:]
            test_label[test_len:test_len+len(test)] = key[-1]
            test_len=test_len+len(test)



    train_shuffle = np.random.permutation(range(train_data.shape[0]))
    train_data = train_data[train_shuffle]
    validate_shuffle = np.random.permutation(range(validation_data.shape[0]))
    validation_data = validation_data[validate_shuffle]
    test_shuffle = np.random.permutation(range(test_data.shape[0]))
    test_data = test_data[test_shuffle]
    train_data=np.double(train_data)/255
    test_data=np.double(test_data)/255
    validation_data=np.double(validation_data)/255
    train_label=train_label[train_shuffle]
    validation_label=validation_label[validate_shuffle]
    test_label = test_label[test_shuffle]

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    print(np.shape(train_data))
        
    # Feature selection
    # Your code here.
    data = np.array(np.vstack((train_data,validation_data,test_data)))
    
    features_count = np.shape(data)[1]

    #col_index = np.arange(features_count)

    count = np.all(data == data[0,:],axis=0)

    data = data[:,~count]

    train_data=data[0:len(train_data),:]
    validation_data=data[len(train_data):len(train_data)+len(validation_data),:]
    test_data=data[len(train_data)+len(validation_data):len(train_data)+len(validation_data)+len(test_data),:]
    print(np.shape(train_data))
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


preprocess()
