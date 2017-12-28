# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 16:04:48 2017

@author: resu
"""

#Part 2： Logistic Regression with a Neural Network mindset
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

'''2 - Overview of the Problem set'''
#x是feature,y是label
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
index = 25  #it can be changed, and see other images. 25 is cat, and 30 is non-cat.
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
m_train = train_set_x_orig.shape[0]#m is the number of training set
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]#num_px is the pixel of the image(the width and the length are the same)

"""
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
"""
train_set_x_flattern=train_set_x_orig.reshape(m_train,-1).T
test_set_x_flattern=test_set_x_orig.reshape(m_test,-1).T

"""
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
"""

"""
One common preprocessing step in machine learning is to center and standardize your dataset,
meaning that you substract the mean of the whole numpy array from each example,
and then divide each example by the standard deviation of the whole numpy array.
But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row of the dataset by 255 (the maximum value of a pixel channel).
"""
train_set_x=train_set_x_flattern/255
test_set_x=test_set_x_flattern/255

'''3 - General Architecture of the learning algorithm'''

'''4 - Building the parts of our algorithm '''
#4.1 - Helper functions
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

#4.2 - Initializing parameters
def initialize(dim):
    w=np.zeros(dim).reshape(dim,1)
    b=0
    assert(isinstance(b, float) or isinstance(b, int))
    return w,b

#4.3 - Forward and Backward propagation
def propagate(w,b,X,Y):
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    cost=(-1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    dw=(1.0/m)*np.dot(X,(A-Y).T)
    db=(1.0/m)*np.sum(((A-Y)))
    cost = np.squeeze(cost)#删除维度为1的那一维
    grads = {"dw": dw,
             "db": db}
    return grads, cost

def optimizie(w,b,X,Y,num_iteration,learning_rate,print_cost=False):
    costs=[]
    for i in range(num_iteration):
        grads,cost=propagate(w,b,X,Y)
        dw=grads["dw"]
        db=grads["db"]
        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i%100==0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        params = {"w": w,
        "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs

def predict(w,b,X):
    m=X.shape[1]
    Ypredict=np.zeros((1,m))
    A= sigmoid(np.dot(w.T,X)+b)
    for i in range(m):
        if A[0,i]>0.5:
            Ypredict[0,i]=1
        else:
            Ypredict[0,i]=0
    return Ypredict


#Merge all functions into a model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w,b=initialize(X_train.shape[0])
    params,grads,costs=optimizie(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    w=params["w"]
    b=params["b"]
    Y_predict_train=predict(w,b,X_train)
    Y_predict_test=predict(w,b,X_test)
    m_train=Y_predict_train.shape[1]
    m_test=Y_predict_test.shape[1]
    print("train accuracy: {} %".format(1-np.sum(np.abs(Y_predict_train-Y_train))/m_train))
    print("test  accuracy: {} %".format(1-np.sum(np.abs(Y_predict_test-Y_test))/m_test))
    d = {"costs": costs,
         "Y_prediction_test": Y_predict_test, 
         "Y_prediction_train" : Y_predict_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)