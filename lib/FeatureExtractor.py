# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 16:49:29 2020

@author: echen2
"""

import numpy as np

import Fetch as df
import fPath

class Extractor1:
    def __init__(self):
        pass
    
    def extract(self, data, feature=None, timestep=None):
        dList = []
        
        try:
            for item in data:
                if feature != None and timestep != None:
                    dList.append(item[timestep][feature])
                elif feature == None:
                    dList.append(item[timestep][:])
                elif timestep == None:
                    dList.append(item[:, feature])
                else:
                    return None
        except IndexError:
            print("Dataset error")
        
        print("Successfully Extracted data")
        return dList

class Extractor:
    def __init__(self):
        pass
    
    def extract(self, data, feature=None, timestep=None):
        empty = True
        dList = None
        
        try:
            for item in data:
                if feature != None and timestep != None:
                    if empty:
                        dList = item[timestep][feature]
                        empty = False
                    else:
                        dList = np.vstack((dList, item[timestep][feature]))
                elif feature == None:
                    if empty:
                        dList = item[timestep][:]
                        empty = False
                    else:
                        dList = np.vstack((dList, item[timestep][:]))
                elif timestep == None:
                    if empty:
                        dList = item[:, feature]
                        empty = False
                    else:
                        dList = np.vstack((dList, item[:, feature]))
                else:
                    return None
        except IndexError:
            print("Dataset error")
        
        print("Successfully Extracted data")
        return dList
    
    
if __name__ == "__main__":
    path = fPath.fPath("C:/Users/echen2/Documents/Git_NAMAC/Baseline_NAMAC_ed/aa01_compressed_2sec/")
    
    F1 = df.Fetch()
    F1.setRoot(path)
    
    dList = []    
    
    for n in range(0, 1023):
        temp = F1.fetchFile("histories_compressed_" + str(n) + ".csv")
        dList.append(temp)
        
    E = Extractor()
    
    feature1 = E.extract(dList, timestep=14)
    