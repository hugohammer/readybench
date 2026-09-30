# -*- coding: utf-8 -*-
"""
Created on May 15, 2023

@author: Meysam Ahangaran
"""
import numpy as np
import pandas as pd

class DataQuality:
    
    def __init__(self):
        self
  
class DataReadinessRecord:
        
    numOfDataQualityMeasures = 4 #number of data quality measures
    featuresList = []   # subset of columns name from master dataset
    rowsIndexes = []    #subset of rows indexes from master dataset
    nullRatioMeasure = 0    # 1 - (null ratio)
    avgPCMeasure = 0    # average Pearson Correlation (1-PC)
    avgSpearmanCorr = 0 #average Spearman Correlation (1-SpearmanCorr)
    outlierUnivariate = 0 #univariate outliers (1-outliers)
    accuracy_classify = 0    #accuracy of classification for the related sub-table
    accuracy_clustering = 0 #accuracy of clustering for the related sub-table
    accuracy_avg = 0 #average accuracy of classify and clustering
    classOverlap = 0 # 1 - classOverlap
    
    
    def __init__(self):
        self
    
    def getTotalQuality(self, precision, avgWeightsFilePath):
        
        df = pd.read_csv(avgWeightsFilePath)
        # df = pd.read_csv("./Output/Runs/Weights/Average_Weights.csv")
        indexes = df.index
        columns = len(df.columns)
        weights = np.zeros(columns, dtype=object)
        for i in range(columns):
            weights[i] = df.xs(indexes[0])[i]
        PC_Coef = weights[0]
        spearman_Coef = weights[1]
        null_Coef = weights[2]
        outlierUnivariate_Coef = weights[3]
        classOverlap_coef = weights[4]
        
        q = ((PC_Coef*self.avgPCMeasure) + (spearman_Coef*self.avgSpearmanCorr) 
             + (null_Coef*self.nullRatioMeasure)
             + (outlierUnivariate_Coef*self.outlierUnivariate)
             + (classOverlap_coef * self.classOverlap))
        return round(q, precision)
        
