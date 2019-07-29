'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pandas as pd
import time


# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return  1.0/(1.0 + np.exp(-1.0*z))

startTime = time.time()
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    
    #print(w1.shape, w2.shape)
    
    # Your code here
    #
    #
    #
    #
    #
    
    # Add bias term
    trainingDataWithBias = np.concatenate((np.ones((training_data.shape[0],1)),training_data),1)
    hiddenLayerSigma = np.dot(trainingDataWithBias, np.transpose(w1))
    hiddenLayerOp = sigmoid(hiddenLayerSigma)
    #print(hiddenLayerSigma.shape, hiddenLayerOp.shape)
    
    hiddenLayerOp = np.concatenate(( np.ones((hiddenLayerOp.shape[0],1)),hiddenLayerOp),1)
    opLayerSigma = np.dot(hiddenLayerOp, np.transpose(w2))
    finalOp = sigmoid(opLayerSigma)
    #print(opLayerSigma.shape, finalOp.shape)
    
    lablesOneHot = np.zeros((len(training_label), n_class))
    
    for i in range(0, len(training_label)):
        lablesOneHot[i][int(training_label[i])] = 1
        
    global tracker
    tracker = lablesOneHot
    #print(training_label.shape)

    error = finalOp - lablesOneHot
    
    #print(error.shape, finalOp.shape)
    #print(np.transpose(error).shape)
    #############################################################################################
    ##  Here it is element wise multiplication because we are supposed to get 
    ##  the y vaulue as single scalar, instead we have used the onehot encoded values
    ##  So we do element wise multiplication of matrix mutiplication. Its VecToVec multiplication
    #############################################################################################
    
    gradientOpToHidden = (error*(finalOp*(1-finalOp)))
    
    #############################################################################################
    ##  We need to take transpose as we have the w2 dimensions as 10X(No of hidden Cells)
    #############################################################################################
    
    #print(error.shape, finalOp.shape)
    gradientOfw2 = np.dot(np.transpose(gradientOpToHidden), hiddenLayerOp)
    #print(gradientOfw2.shape)
    
    #print(gradientOpToHidden.shape, w2.shape)
    gradientOfHiddenOp = np.dot(gradientOpToHidden, w2)
    #print(gradientOfHiddenOp.shape)
    
    #print(gradientOfHiddenOp.shape, hiddenLayerOp.shape)
    ## Again we need to do element wise multiplication
    gradientOfhiddenIp = gradientOfHiddenOp*hiddenLayerOp*(1-hiddenLayerOp)
    #print(gradientOfhiddenIp.shape)

    gradientOfw1 = np.dot(np.transpose(gradientOfhiddenIp), trainingDataWithBias)
    #print(gradientOfhiddenIp.shape, trainingDataWithBias.shape, gradientOfw1.shape)
    
    ## 
    negativeLogLikeliHood = -np.sum((lablesOneHot*np.log(finalOp)) + ((1-lablesOneHot)*np.log(1- finalOp)))
    obj_val = (negativeLogLikeliHood + (lambdaval/2)*(np.sum(w1**2) + np.sum(w2**2)))/training_data.shape[0]
    
    grad_w1 = (gradientOfw1[1:n_hidden+1,] + (lambdaval * w1))/training_data.shape[0]
    grad_w2 = (gradientOfw2 + (lambdaval * w2))/training_data.shape[0]
    
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    #print(obj_val)
    return (obj_val, obj_grad)
# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    labels = np.array([])
    # Your code here
    
    dataWithBias = np.concatenate(( np.ones((data.shape[0], 1)),data),1)
    
    hiddenLayerSigma = np.dot(dataWithBias, np.transpose(w1))
    hiddenLayerOp = sigmoid(hiddenLayerSigma)
    
    hiddenLayerOp = np.concatenate(( np.ones((hiddenLayerOp.shape[0],1)),hiddenLayerOp),1)
    opLayerSigma = np.dot(hiddenLayerOp, np.transpose(w2))
    finalOp = sigmoid(opLayerSigma)
    
    global tracker
    tracker = finalOp
    labels = np.argmax(finalOp,axis=1)
    

    return labels
# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)


#noOfHiddenNodes = [4, 8, 12, 16, 20]
#lambas = [0,10,20,30,40,50,60]
#
#for lamba in lambas:
#    for hiddenNodes in noOfHiddenNodes:
        
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval=0
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
endTime = time.time()
print("Time take for 1 hidden layer on FaceA data is")
print(endTime-startTime)

#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
