# -*- coding: utf-8 -*-
"""
Created on May 15, 2023

@author: Meysam Ahangaran
"""
import numpy as np
from scipy import stats


class Outlier:
    def __init__(self):
        self
    
    #MAD method (univariate outlier)
    def mad_method(self,df, variable_name):
        #Takes two parameters: dataframe & variable of interest as string
        columns = df.columns
        med = np.median(df, axis = 0)
        mad = np.abs(stats.median_abs_deviation(df))
        threshold = 3
        outlier = []
        index=0
        for item in range(len(columns)):
            if columns[item] == variable_name:
                index == item
        for i, v in enumerate(df.loc[:,variable_name]):
            t = abs(v-med[index])/mad[index]
            if t > threshold:
                outlier.append(i)
            else:
                continue
        numOfOutliers = len(outlier)
        return outlier, numOfOutliers
    
