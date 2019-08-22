import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pywt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import random
from scipy.spatial import distance
import math
from typing import Union

DataSources = Union[str,pd.Series,np.ndarray]

class TemplateError():
    pass

class DWTTemplate():   
    
    def __init__(self, template: np.ndarray, idxs = (0,0), start_level = 4):
        self.__level = start_level
        self.template = template
        self.template_norm = template/np.sum(template)
        self.idxs = idxs
        self.strart_level = start_level
        coeffs = pywt.wavedec(self.template_norm, 'haar', level = start_level)   
        self.ca, self.cd = coeffs[0],coeffs[1:]

    def down(self)->None:
        """
        Reduces the wavelet transform level of the Series Piece.
        """
        self.__level -= 1
        if self.__level == 0:
            raise TemplateError("The wavelet transform is too low. The level cannot be lower than 1.")
        coeffs = pywt.wavedec(self.template_norm, 'haar', level = self.__level)   
        self.ca, self.cd = coeffs[0],coeffs[1:]  


class DWTTemplates():       
    
    def __init__(self, series:DataSources, start_level = 4, pattern_len = 120,
                 t_count = 10000, temp_type = 'shuffle', acceleration = True):
        self.series = self.__init_series(series)
             
        self.current_level = start_level  
        self.all_patters = False
        self.__ixs = None
        self._templates =  None      
        self._acceleration = acceleration
#         self._templates = self.__init_templates(temp_type, start_level, pattern_len, t_count)
        
        
    def set_options(self, start_level = 4, pattern_len = 120, t_count = 10000, temp_type = 'shuffle'):
        self._templates = self.__init_templates(temp_type, start_level, pattern_len, t_count)       

    def __init_series(self,series:DataSources)->np.ndarray:        
        if isinstance(series, str):
            return pd.read_csv(series)
        if isinstance(series,  pd.Series):
            return series.values
        if isinstance(series,  np.ndarray):
            return series
        return None      
    
    def __init_templates(self, temp_type:str, start_level:int, pattern_len:int, t_count = None)->np.ndarray:
        """
        Split time series to Series Pieces.
        """
        piece_len = pattern_len
        slices = []        
        ixs = []
        ixs = np.asarray(list(range(0,len(self.series)- piece_len)))
        if temp_type == 'shuffle':
            np.random.shuffle(ixs)
            self.__ixs = ixs[:t_count] 
        elif temp_type == 'half_step':
            step = pattern_len//2
            self.__ixs = ixs[::step]
        elif temp_type == 'third_step':
            step = pattern_len//3
            self.__ixs = ixs[::step]        
        for i in range(0, len(self.series)- pattern_len):
            part = self.series[i:i+pattern_len]
            part = DWTTemplate(part, start_level = start_level, idxs = (i, i+pattern_len))
            slices.append(part)
        slices = np.asarray(slices)
        return slices
    
    @property
    def cur_templates(self):
        """
        List all pieces of temporary data in a -Template- format.
        """
        return self._templates[self.__ixs]
    
    @property
    def templates(self):
        """
        List of all aggregated pieces of temporary format np.ndarray data.
        """
        itemplates = []
        if self.all_patters == False:
            T = self._templates[self.__ixs]
        else:
            T = self._templates
        for t in T:
            itemplates.append(t.ca)
        return  np.asarray(itemplates)
    
    @property
    def original_templates(self):
        """
        List of all original pieces of temporary format np.ndarray data.
        """
        itemplates = []
        if self.all_patters == False:
            T = self._templates[self.__ixs]
        else:
            T = self._templates
        for t in T:
            itemplates.append(t.template_norm)
        return  np.asarray(itemplates)
    
    def down(self):
        if self._acceleration == True:
            for i in self.__ixs:
                    self._templates[i].down()
        else:                     
            for i,t in enumerate(self._templates):
                    self._templates[i].down()
        return 
    
if __name__ == "__main__":
    data = pd.read_csv("data.csv", skiprows = 5, header = None, sep="\s+")
    train_data = data[4]
    templates = DWTTemplates(train_data, pattern_len = 96, start_level = 4)
    t = templates.templates
    to = templates.original_templates
    
    
#    n_clusters = 30
#    kmeans = KMeans(n_clusters=120)
#    y_kmeans = kmeans.fit_predict(t)
#    
#    mask = np.where(y_kmeans == 5)[0][:-1]
#    t1 = t[mask]
#    to1 = to[mask]
#    sqeres_plot(t1, size = (4,4))
#    sqeres_plot(to1, size = (4,4))