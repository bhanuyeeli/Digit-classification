import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pandas as pd
import time
import matplotlib.pyplot as plt



def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  1.0/(1.0 + np.exp(-1.0*z))# your code here


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

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    
    trainRows = 0
    testRows = 0
    totalRows = 0
    
    for key1 in list(mat.keys()):
        if ("train" in key1 or "test" in key1):
            
            totalRows += mat[key1].shape[0]
            if "train" in key1:
                trainRows += mat[key1].shape[0]
            else:
                testRows += mat[key1].shape[0]
    
    #print(trainRows, testRows, totalRows)
    noOfCols = mat['train1'].shape[1]
    
    totalTrainData = np.zeros((trainRows, noOfCols))
    totalLables = np.zeros((trainRows, 1))
    
    validationRows = 10000
    finalTrainRows = trainRows-validationRows
    
    train_data = np.zeros((finalTrainRows, noOfCols))
    train_label = np.zeros((finalTrainRows, 1))
    validation_data = np.zeros((validationRows, noOfCols))
    validation_label = np.zeros((validationRows, 1))
    test_data = np.zeros((testRows, noOfCols))
    test_label = np.zeros((testRows, 1))
    
    trainRowsSoFar = 0
    testRowsSoFar = 0
    for key1 in list(mat.keys()):
        if ("train" in key1):
           rows = mat[key1].shape[0]
           ## Feature Scaling
           totalTrainData[trainRowsSoFar:trainRowsSoFar+rows, ] = mat[key1]/255
           totalLables[trainRowsSoFar:trainRowsSoFar+rows, ] = np.array(np.repeat(int(key1.split('train')[1]), rows)).reshape(rows,1)
           trainRowsSoFar += rows
        elif ("test" in key1):
           rows = mat[key1].shape[0]
           ## Feature Scaling
           test_data[testRowsSoFar:testRowsSoFar+rows, ] = mat[key1]/255
           test_label[testRowsSoFar:testRowsSoFar+rows, ] = np.array(np.repeat(int(key1.split('test')[1]), rows)).reshape(rows,1)
           testRowsSoFar += rows
    
    trainIndexes = np.random.choice(range(0,trainRows), finalTrainRows, replace=False)
    validationIndexes = list(set(range(0,trainRows)) - set(trainIndexes))
    
    train_data = totalTrainData[trainIndexes,]
    train_label = totalLables[trainIndexes,]
    validation_data = totalTrainData[validationIndexes, ]
    validation_label = totalLables[validationIndexes,]
    
    # Did feature scaling by dividing by 255. Actual formula for scaling is (X-Xmin)/(Xmax-Xmin) 
    # Here we already have the Xmin as 0 and Xmax as 255 hence we get X/255

    # Feature selection
    # Your code here.
    
    deleteThreshold = 10
    
    deletColIndexes = []
    for col in range(0, noOfCols):
        if(list(train_data[:,col]).count(0) >= (finalTrainRows - deleteThreshold)):
           deletColIndexes.append(col) 
    
    train_data = np.delete(train_data, deletColIndexes, axis = 1)
    validation_data = np.delete(validation_data, deletColIndexes, axis = 1)
    test_data = np.delete(test_data, deletColIndexes, axis = 1)
    
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

tracker = 0
def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

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


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

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


"""**************Neural Network Script Starts here********************************"""
colNames = ['Lambda','Hidden','TrainAccuracy','ValidationAccuracy','TestAccuracy','Time']
observations = pd.DataFrame(columns = colNames)

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

noOfHiddenNodes = [4, 8, 12, 16, 20, 35, 60]
lambas = [0,10,20,30,40,50,60]
Train_accuracy=[]
Val_accuracy=[]
Test_accuracy=[]

#noOfHiddenNodes = [50]
#lambas = [0]

rowCounter = 0
#  Train Neural Network
for lamba in lambas:
    for hiddenNodes in noOfHiddenNodes:
# set the number of nodes in input unit (not including bias unit)
        
        startTime = time.time()
        
        n_input = train_data.shape[1]
        
        # set the number of nodes in hidden unit (not including bias unit)
        n_hidden = hiddenNodes
        
        # set the number of nodes in output unit
        n_class = 10
        
        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)
        
        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
        
        # set the regularization hyper-parameter
        lambdaval = lamba
        
        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
        
        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
        
        opts = {'maxiter': 50}  # Preferred value.
        
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
        
        endTime = time.time()
        runningTime = endTime-startTime
        # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        # and nnObjGradient. Check documentation for this function before you proceed.
        # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)
        
        
        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
        
        # Test the computed parameters
        
        predicted_label = nnPredict(w1, w2, train_data)
        
        def GetAccuracy(predcited, actual):
            
            count = 0
            for i in range(0, len(predcited)):
                if(int(predcited[i]) == int(actual[i])):
                    count += 1
                    
            return (float(count)/float(len(predcited)))*100
            
        
        # find the accuracy on Training Dataset
        
        trainAccuracy = GetAccuracy(predicted_label, train_label)
        
        print('\n Training set Accuracy:' + str(trainAccuracy) + '%')
        Train_accuracy.append([GetAccuracy(predicted_label, train_label),lamba,hiddenNodes])
        
        predicted_label = nnPredict(w1, w2, validation_data)
        
        # find the accuracy on Validation Dataset
        
        validationAccuracy = GetAccuracy(predicted_label, validation_label)
        
        print('\n Validation set Accuracy:' + str(validationAccuracy) + '%')
        Val_accuracy.append([GetAccuracy(predicted_label, validation_label),lamba,hiddenNodes])
        
        predicted_label = nnPredict(w1, w2, test_data)
        
        # find the accuracy on Validation Dataset
        testAccuracy = GetAccuracy(predicted_label, test_label)
        
        print('\n Test set Accuracy:' + str(testAccuracy) + '%')
        Test_accuracy.append([GetAccuracy(predicted_label, test_label),lamba,hiddenNodes])
        
        #collections.Counter(predicted_label)
        
        observations.loc[rowCounter] = [lamba,hiddenNodes,trainAccuracy,validationAccuracy,testAccuracy,runningTime]
        
        rowCounter += 1
        

avgTestAccuracyByLambda = observations.groupby("Lambda").agg({"TestAccuracy" : 'mean'})
#bars = avgTestAccuracyByLambda.index
#y_pos = np.arange(len(bars))
plt.bar(avgTestAccuracyByLambda.index, avgTestAccuracyByLambda['TestAccuracy'].values, color=('green', 'green', 'green', 'red', 'red', 'yellow', 'red'))
#plt.xticks(y_pos, bars)
plt.title('Average Test Accuracy by Lambda')
plt.xlabel('Lambda')
plt.ylabel('Test Accuracy %')
plt.savefig('nnscript_plot1.png')
plt.show()

avgTestAccuracyByHiddenLayers = observations.groupby("Hidden").agg({"TestAccuracy" : 'mean'})
plt.bar(avgTestAccuracyByHiddenLayers.index, avgTestAccuracyByHiddenLayers['TestAccuracy'].values, color=('red', 'yellow', 'yellow', 'yellow', 'green', 'yellow', 'green'))
plt.title('Average Test Accuracy by Hidden Nodes')
plt.xlabel('Hidden nodes')
plt.ylabel('Test Accuracy %')
#plt.savefig('nnscript_plot2.png')
plt.show()

# Make some fake data.
a =  np.arange(0, 49, 1)

# Create plots with pre-defined labels.
#fig, ax = plt.subplots()
#ax.plot(a, Train_accuracy , 'k--', label='Train accuracy')
#ax.plot(a, Val_accuracy, 'k:', label='Validation accuracy')
#ax.plot(a, Test_accuracy[], 'k', label='Test accuracy')

#legend = ax.legend(loc='lower center', shadow=True, fontsize='x-small')

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C0')
#plt.savefig('nnscript_plot3.png')
#plt.show()

#noOfHiddenNodes_new=7*noOfHiddenNodes
#for i in noOfHiddenNodes[i]  :
#    lambas_new=7*lambas

# Exporting Observations to csv
observations.to_csv(r'nnscript_accuracies.csv')

# for i in observations[np.where]:

#observations[np.where(observations['Lambda']==0)].plot(x='Hidden',y='TestAccuracy')

#df.loc[(df.a == 10) & (df.b == 20), ['x','y']].plot(title='a: 10, b: 20')

#observations.loc[(observations.Lambda == 0), ['Hidden','TestAccuracy']].plot(title='Lambda = 0')
#observations.loc[(observations.Lambda == 0),['Hidden','TestAccuracy']].plot(title='Lambda = 0')


for i in lambas:
    observations.loc[(observations.Lambda == i)].plot(x='Hidden',y='TestAccuracy')
    plt.title('Lambda = %i' %i)
    plt.xlabel('Hidden nodes')
    plt.ylabel('Test Accuracy %')
    plt.savefig('nnscript_accu_plot%i.png'%i)
plt.show()

Avg_training_time = observations.groupby("Hidden").agg({"Time" : 'mean'})
