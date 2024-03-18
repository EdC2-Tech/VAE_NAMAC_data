# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 11:43:42 2020

@author: echen2
"""

import numpy as np
import pprint as pp


class Fetch:
    '''
    Retrieves the csv file located at the path
    
    Keyword Variables:
        root       -- fPath object, folder path to datasets 

    '''
    
    def __init__ (self):
        self.root = None;
        
    def setRoot(self, fPath):
        '''
        Sets the folder path to datasets

        Parameters
        ----------
        fPath : fPath
            Folder path

        Returns
        -------
        None.

        '''
        self.root = fPath.root
        
    def fetchFile(self, filename):
        '''
        Gets the file. Assumes that the file has the root previously assigned.

        Parameters
        ----------
        filename : String
            Name of the csv file imported

        Returns
        -------
        dArray : ndarray
            array retrieved from the csv file

        '''
        path = self.root + filename
        dArray = np.genfromtxt(path, delimiter=',', skip_header=True)    
        
        return dArray
        
if __name__ == "__main__":
    import fPath
    path1 = fPath.fPath("C:/Users/echen2/Documents/Git_NAMAC/Baseline_NAMAC_ed/aa01_histories/histories_short_ref/")
    path2 = fPath.fPath("C:/Users/echen2/Documents/Git_NAMAC/Baseline_NAMAC_ed/aa01_compressed_1sec/")
    
    F1 = Fetch()
    F1.setRoot(path1)
    
    F2 = Fetch()
    F2.setRoot(path2)
    
    dArray = None
    
    for n in range(0,1):
         dArray = F1.fetchFile("histories_short_print_" + str(n) + ".csv")

        