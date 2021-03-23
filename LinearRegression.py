#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA

from collections import defaultdict

import unittest


# In[2]:



class LinearRegression():
    
        
    def __init__(self,X_train,Y_train):
        self.X_train=np.append([[1]]*len(X_train),X_train,1)
        self.Y_train=Y_train
        self.fittedLine=self.findBestFitLine()
        
    def fit(self,X_train,Y_train):
        self.X_train=np.append([[1]]*len(X_train), X_train,1)
        self.Y_train=Y_train
        self.fittedLine=self.findBestFitLine()
        
    def findBestFitLine(self):
        A=np.array([])
        Y=np.array([])
        for i in range(self.X_train.shape[1]):
            A=np.append(A,[(self.X_train[...,i].T@self.X_train)])
            Y=np.append(Y,(self.X_train[...,i].T@self.Y_train))
        A=A.reshape((self.X_train.shape[1],self.X_train.shape[1]))
        Y=Y.reshape((self.X_train.shape[1],1))
        return LA.inv(A)@Y
    
    def predict(self,newPoint):
        point=np.append([[1]],newPoint,1)
        return point@self.fittedLine
    
    def predict_batch(self,batchPoint):
        return [self.predict(point) for point in batchPoint]


    


# In[3]:


class Test(unittest.TestCase):
    
    # This is a variable to generate normal train set with train_size size
    #increasing or decreasing it may effect the test
    train_size=5000
    
    # batchsize is used to test the batch methods
    batchsize=100
    
    # This is a variable to generate huge train set with train_size size
    #increasing this will effect the time of the tests
    efficiency_train_size=100000
    
    def generate_random_noraml_point(self,count,pointCount):
        point=[[(i+count)] for i in range(pointCount)]
        x_train=[[i] for i in range(count)]
        y_train=[[i] for i in range(count)]+np.random.randn(count, 1)
        return point,x_train,y_train 
    
    def test_fitLine(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        model=LinearRegression(np.array(x_train),np.array(y_train))
        bestLine=model.findBestFitLine()
        assert LA.norm(bestLine-[[0],[1]]) < 0.06
        
    def test_predict(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        model=LinearRegression(np.array(x_train),np.array(y_train))
        y=model.predict(point)
        assert LA.norm(y-point) < 0.06
    
        
        
if __name__== '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)


# In[ ]:




