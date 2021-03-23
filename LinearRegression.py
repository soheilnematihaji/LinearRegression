#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA

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
        A=self.X_train.T@self.X_train
        Y=self.X_train.T@self.Y_train
        return LA.pinv(A)@Y
    
    def predict(self,newPoint):
        point=np.append([[1]],newPoint,1)
        return point@self.fittedLine
    
    def predict_batch(self,batchPoint):
        batchPoint=np.append([[1]]*len(batchPoint),batchPoint,1)
        return batchPoint@self.fittedLine 


    


# In[3]:


class Test(unittest.TestCase):
    
    # This is a variable to generate normal train set with train_size size
    #increasing or decreasing it may effect the test
    train_size=5000
    
    # batchsize is used to test the batch methods
    batchsize=100
    
    # This is a variable to generate huge train set with train_size size
    #increasing this will effect the time of the tests
    efficiency_train_size=1000000
    
    batchsize_eff=100000
    def generate_random_noraml_point(self,count,pointCount):
        point=[[(i+count)] for i in range(pointCount)]
        x_train=[[i] for i in range(count)]
        y_train=[[i] for i in range(count)]+np.random.randn(count, 1)
        return point,x_train,y_train 
    
    def test_fitLine(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        model=LinearRegression(np.array(x_train),np.array(y_train))
        bestLine=model.findBestFitLine()
        assert LA.norm(bestLine-[[0],[1]]) < 0.1
        
    def test_predict(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        model=LinearRegression(np.array(x_train),np.array(y_train))
        y=model.predict(point)
        assert LA.norm(y-point) < 0.06
        
    def test_predictBatch(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,self.batchsize)
        model=LinearRegression(np.array(x_train),np.array(y_train))
        y=model.predict_batch(np.array(point))
        for i in range(len(point)):
            assert LA.norm(y[i]-point[i]) < 0.06
            
    def test_predictBatch_efficiency(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.efficiency_train_size,self.batchsize_eff)
        model=LinearRegression(np.array(x_train),np.array(y_train))
        y=model.predict_batch(np.array(point))
    
        
        
if __name__== '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

