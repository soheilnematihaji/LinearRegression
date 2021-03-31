#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy import linalg as LA
import GradientDescent
import NormalScalar
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
        
    def findBestFitLine_using_gd(self,learning_rate=0.01):
        
        #Normalizing the inpurt before applying Gradient Descent
        x_temp_toScale=np.concatenate((self.X_train[:,1:],self.Y_train),1)
        normalScalar= NormalScalar.NormalScalar()
        normaled_val=normalScalar.fit_transform(x_temp_toScale)
        X_train=np.append([[1]]*len(normaled_val),normaled_val[:,:-1],1)
        Y_train=normaled_val[:,-1:]
        
        #Applying Gradient Descent on cost funtion J
        (m,n)=X_train.shape
        J= lambda theta : ((X_train@theta.reshape(n,1)-Y_train).T@(X_train@theta.reshape(n,1)-Y_train))[0][0]/(2*m)
        gd=GradientDescent.GradientDescent()
        theta= gd.gradientDescent(J, np.array([0 for i in range(n)]),learning_rate=learning_rate,delta_val=0.000001, iterations=10000)
        
        # Inverse Scaling the value of theta
        theta[0]=normalScalar.std[-1]*(theta[0]-(sum(normalScalar.mean[:-1] *(theta[1:]/normalScalar.std[:-1]) )))+normalScalar.mean[-1]
        theta[1:]=(theta[1:]/normalScalar.std[:-1])*normalScalar.std[-1]
        
        return theta
        
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
    efficiency_train_size=100000
    
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
        
    def test_fitLine_theta0_Nonezero(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        model=LinearRegression(np.array(x_train),np.array(1+y_train))
        bestLine=model.findBestFitLine()
        assert LA.norm(bestLine-[[1],[1]]) < 0.1
        
    def test_findBestFitLine_using_gd(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        model=LinearRegression(np.array(x_train),np.array(y_train))
        bestLine=model.findBestFitLine_using_gd()
        assert LA.norm(bestLine-[0,1]) < 0.2
        
    def test_predict(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,1)
        model=LinearRegression(np.array(x_train),np.array(y_train))
        y=model.predict(point)
        assert LA.norm(y-point) < 0.1
        
    def test_predictBatch(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.train_size,self.batchsize)
        model=LinearRegression(np.array(x_train),np.array(y_train))
        y=model.predict_batch(np.array(point))
        for i in range(len(point)):
            assert LA.norm(y[i]-point[i]) < 0.1
            
    def test_predictBatch_efficiency(self):
        point,x_train,y_train=self.generate_random_noraml_point(self.efficiency_train_size,self.batchsize_eff)
        model=LinearRegression(np.array(x_train),np.array(y_train))
        y=model.predict_batch(np.array(point))
    
        
        
if __name__== '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

