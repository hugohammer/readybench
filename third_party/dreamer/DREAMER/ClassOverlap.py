# -*- coding: utf-8 -*-
"""
Created on May 15, 2023

@author: Meysam Ahangaran
"""

import numpy as np
import numba


class ClassOverlap:
    
    k = 7   #kNN parameter
    tetha = 3   #threshold of R-Value
 
    def __init__(self):
        self
    
    
    #get number of samples of each class. Last column is always the class variable.
    def getNumofSamples(self, data, className):
        rows = len(data)
        cols = len(data[0])
        ctr = 0
        for i in range(rows):
            if data[i][cols-1] == className:
                ctr = ctr + 1
        
        return ctr
    
    #get num of kNN for a class in the other class
    def getRvalueClass(self, data, className):
        
        rows = len(data)
        cols = len(data[0])
        data_num = np.array(data[0:(rows),0:(cols-1)], dtype=np.float32)
        data_class = np.array(data[:,(cols-1)], dtype=np.str0)
        
        kNN_ctr = 0 #number of samples in the className with overlapping property
        
        for i in range(rows):
            if data[i][cols-1] == className:
               kNN_otherClass = self.getKnnOtherClass(data_num, data_class, className, i)
               if kNN_otherClass > self.tetha:
                   kNN_ctr = kNN_ctr + 1
        
        return kNN_ctr
        
                
    #get num of Knn of a idx in other class of className
    def getKnnOtherClass(self, data_num, data_class, className, idx):
        
        kNN_otherClass = 0
        kNN_arrIndexes = _getKnnOtherClass(data_num, idx, self.k)
        for i in range(self.k):
            
            if data_class[kNN_arrIndexes[i]] != className:
                kNN_otherClass = kNN_otherClass + 1
        
        return kNN_otherClass
    
    
    
    # calculate the R-Value for the input data (last column is target)
    def getRvalue(self, data):
        
        classNames_list = getTargetArray(data)
        numOfClasses = len(classNames_list)
        totalSamples = 0
        total_RValue = 0
       
        
        for i in range(numOfClasses):
            samples = self.getNumofSamples(data, classNames_list[i])
            totalSamples = totalSamples + samples
        
        for j in range(numOfClasses):
            RVal = self.getRvalueClass(data, classNames_list[j])
            total_RValue = total_RValue + RVal
        
        R_value = total_RValue / totalSamples
        return R_value
    

#get array of class names
def getTargetArray(data):
        rows = len(data)
        cols = len(data[0])
        classNames = [] #list of all class names
        
        for i in range(rows):
            target = data[i, (cols-1)]
            if target in classNames:
                continue
            classNames.append(target)
        return classNames


@numba.jit(nopython=True)
def _getKnnOtherClass(data, idx, k):
        
        rows = len(data)
        cols = len(data[0])
        distArr = np.zeros(rows, dtype=np.float32) #distances of samples to the idx sample
        kNN_arrIndexes = np.zeros(k, dtype=np.int32) #array of kNN indexess
        
        for i in range(rows):
            v1 = data[i, :]
            v2 = data[idx, :]                
            distArr[i] = np.sqrt(np.sum((v1 - v2) ** 2))
            
        
        for i in range(k):           
            for k in range(rows):
                if ~(k in kNN_arrIndexes):
                    minIdx = k
                    break
              
            for j in range(rows):
                if (j==idx or j in kNN_arrIndexes):  #current index element is discarded
                    continue
                if distArr[j] <= distArr[minIdx]:
                    minIdx = j
            kNN_arrIndexes[i] = minIdx                                                           
        
        return kNN_arrIndexes

    
