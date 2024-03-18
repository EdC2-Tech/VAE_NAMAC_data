# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:49:54 2022

@author: ChEdw
"""
import numpy as np
import sys

import fPath
import Fetch as DF
import FeatureExtractor as FE

sys.path.append("./lib")
       
root = "C:/Users/ChEdw/Desktop/VAE_NAMAC_data/Data_private/"

def initialize_any(dname):
    """
    Set path to the dataset folder
    MODIFY TO USE DIFFERENT DATASET
    """
    path0      = fPath.fPath(root + dname)

    '''
    Initialize data retrival objects for each separate dataset path
    '''
    F0 = DF.Fetch()
    F0.setRoot(path0) # Original, no compression

    '''
    Initilize feature extractor. Extracts data by columns or rows
    '''
    extractor = FE.Extractor() 

    return path0, F0

def load_any(ep_begin, ep_end, F0, interval=1, title=''):
    '''
    Load all unformated uncompressed data

    Parameters
    ----------
    ep_begin : int
        Beginning episode.
    ep_end : int
        Last episode.
    F0 : fPath
        File path object.
    interval : int, optional
        Skipping value. The default is 1.

    Returns
    -------
    raw_input_test0 : ndArray
        Model input as array.
    raw_output_test0 : ndArray
        Intended model output as array.
    time : ndArray
        Vector of time stamps.
    dList0 : ndArray
        Metadata; contains all information.
    '''
    dList0 = []
    start = ep_begin
    end = ep_end

    #Get data
    for n in range(start, end):
        if n%interval==0:
            temp = F0.fetchFile(title + str(n) + ".csv")
            dList0.append(temp)
            
    all_list = np.vstack(dList0)
    
    time     = all_list[:,0]
    features = all_list[:,0:]    
    
    return time, features, dList0
    