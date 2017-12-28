# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 18:39:41 2017

@author: resu
"""

import numpy as np

#1.1 - sigmoid function, np.exp()
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

#1.2 - Sigmoid gradient
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))#隐函数求导

#1.3 - Reshaping arrays
def image2vector(image):
    v=image.reshape(image.shape[0]*image.shape[1]*image.shape[2],1)
    return v

#1.4 - Normalizing rows
def normalize(x):  #这个比较高级了用x除以x的模不是除以和
    x_norm=np.linalg.norm(x,axis=1,keepdims = True)
    x=x/x_norm
    return x

#1.5 - Broadcasting and the softmax function
def softmax(x):
    x_exp=np.exp(x)
    m,n=x_exp.shape
    x_sum=np.sum(x_exp,axis=1, keepdims = True)
    s=x_exp/x_sum
    return s

#2) Vectorization
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT OF VECTORS IMPLEMENTATION ###
def classicdot(x1,x2):
    dot = 0
    for i in range(len(x1)):
        dot+= x1[i]*x2[i]
    return dot

### CLASSIC OUTER PRODUCT IMPLEMENTATION ###
def classic_outer_product(x1,x2):
    outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
    for i in range(len(x1)):
        for j in range(len(x2)):
            outer[i,j] = x1[i]*x2[j]
    return outer
### CLASSIC ELEMENTWISE IMPLEMENTATION ###
def classic_elementwise(x1,x2):
    mul = np.zeros(len(x1))
    for i in range(len(x1)):    
        mul[i] = x1[i]*x2[i]
    return mul
### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
def classic_general_dot_product(x1,x2):
    W = np.random.rand(3,len(x1)) # Random 3*len(x1) numpy array
    gdot = np.zeros(W.shape[0])
    for i in range(W.shape[0]):
        for j in range(len(x1)):
            gdot[i] += W[i,j]*x1[j]
    return gdot

#2.1 Implement the L1 and L2 loss functions
def l1(yhat,y):
    loss=np.sum(np.abs(y-yhat))
    return loss

def l2(yhat,y):
    loss=np.sum(np.power((y-yhat),2))
    return loss

