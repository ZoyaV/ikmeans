import pandas as pd
import numpy as np
from typing import Union
import pywt
from sklearn.cluster import KMeans

import sys
sys.path.append('../../ikmeans')
from dwt_templates import DWTTemplates
from _ikmeans import *
from ikmeans_error import IKMeansError  

DataSources = Union[str,pd.Series,np.ndarray]

class IKMeans():
    
    def __init__(self, data:DataSources, start_n_clusters:int, down_iterations = 2, init = None):
        self.start_n_clusters = start_n_clusters
        self.down_iterations = down_iterations
        self.init = init
        self.templates_ = DWTTemplates(data)
        self.kmeans = KMeans(n_clusters = self.start_n_clusters) 
        self.ik_labels = None  
        
        self.__init = False
    
    def fit(self)->None:
        """
        Initial clustering
        """
        temp_agrig = self.templates_.templates
        self.ik_labels = self.kmeans.fit_predict(temp_agrig)
        self.init = self.kmeans.cluster_centers_
        
        self.__init = True
        
    @property
    def labels(self)->np.ndarray:
        """
        The label of each pattern
        """
        return self.ik_labels
        
    def next_step(self)->None:
        """
        Clusters data at the next level of wavelet transform
        """
        if  self.__init == False:
            raise IKMeansError("Initial clustering failed! Call the .fit Method!")
        self.init = new_centers(self.kmeans.cluster_centers_)
        self.templates_.down()
        temp_agrig = self.templates_.templates
        if self.init is not None:
            CLASTER_COUNT = len(self.init)
            self.kmeans = KMeans(n_clusters=CLASTER_COUNT, init = self.init)
        if len(self.init[0])!=len(temp_agrig[0]):
            raise IKMeansError("""
                     The size of the cluster centers and the data for clustering must match.
                    The size of the centers T is when the data size is T. I have no idea where this error 
                    comes from, but it is solved by increasing the length of the pattern.""")
        self.ik_labels = self.kmeans.fit_predict(temp_agrig)  
        self.init = self.kmeans.cluster_centers_
            
    def most_popular_clasters(self)->list:
        """
        Return sorted counts of every labels
        [[id_of_patter, count_of_pattern],...]
        """
        unique, counts = np.unique(self.ik_labels, return_counts=True)
        all_counts = sorted(list(zip(unique, counts)), key = lambda t:t[1])[::-1]
        return all_counts

