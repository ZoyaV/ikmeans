# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 18:45:05 2019

@author: Zoya
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import pywt


def optimal_size(count:int)->tuple:
    """
    Return optimal size to plot grafics
    IN:
        count - the count of grafics
    OUT:
        w, h - width and height counts of grafics
    """
    p = count**0.5
    width = math.ceil(p)
    height = math.floor(p)   
    return width, height

def sqeres_plot(templates, size = None, save = False, name = 'default'):
    """
    Plot graphics of one of templates
    IN:
        templates - np.ndarray with patterns
    """
    if size == None:
        w,h = optimal_size(len(templates))
    else:
        w,h = size
    plt.figure(figsize = (w*1.67,h*1.67))
    k = 0
    for i in range(w):
        for j in range(h):
            try:
                plt.subplot(h,w,k+1)
                plt.plot(templates[k])
                plt.title(str(k))
                k+=1
            except Exception as e:
                break
                pass
    plt.tight_layout()
    if save:
        name ='%s.png'%name
        plt.savefig(name)
    plt.show()  
    
def pattern_plot_agr(data, templates, level:int, zoom_range = [0,-1],
                     color = 'b', end = False, save = False, name = "figure"):
    """
    Plot templates on time series
    IN:
         data - np.ndarray with patterns;
         templates - patterns as np.ndarray with DWTTemplate type;
         level - DWT level in what you want represent data;
         zoom_range - list of type[start_slice_id: end_slice_id]; zooming data in that range;
         color - the color of plot;
         end -  True - add point in end and start of graphics;
         save - True - save figure;
         name - name of saved file.
    OUT:
         None
    """
    plt.figure(figsize=(12,2))
    alpha = 0.2
    linestyle = '-'
    linewidth = 1.5
#     templates = templates[mask]
    
    coeffs = pywt.wavedec(data, 'haar', level = level)   
    ca, cd = coeffs[0],coeffs[1:] 
    
    d = 2**level
    
    if zoom_range[1] == -1:
        zoom_range[1] = len(ca)
    y = ca[zoom_range[0]: zoom_range[1]]
    x = range(zoom_range[0], zoom_range[1])
    plt.plot(x,y, color = color,
                             alpha = alpha,
                             linewidth = linewidth,
                             linestyle = linestyle)
    
    for t in templates:
        coords = t.idxs
        coords_agr = coords[0]//d, coords[1]//d
        if coords_agr[0]>zoom_range[0] and coords_agr[1]<zoom_range[1]:
            x = range(coords_agr[0], coords_agr[1])
            y = ca[coords_agr[0]: coords_agr[1]]
            plt.plot(x,y, color = color, linewidth = 2)
            if end:
                plt.scatter(x[0], y[0], color = 'r')
                plt.scatter(x[-1], y[-1], color = 'r')                
    if save == True:
        plt.savefig(name)
                
def pattern_plot(templates, series, zoom_range = [0,-1], color = 'r', 
                 end = True, save = False, name = "figure"):    
    """
    Plot templates on time series
    IN:
        data - np.ndarray with patterns;
        templates - patterns as np.ndarray with DWTTemplate type;
        zoom_range - list of type[start_slice_id: end_slice_id]; zooming data in that range;
        color - the color of plot;
        end -  True - add point in end and start of graphics;
        save - True - save figure;
        name - name of saved file.
    OUT:
        None
    """
    plt.figure(figsize=(12,2))
    alpha = 0.2
    linestyle = '-'
    linewidth = 1
#     templates = templates[mask]
    if zoom_range[1] == -1:
        zoom_range[1] = len(series)
    y = series[zoom_range[0]: zoom_range[1]]
    x = range(zoom_range[0], zoom_range[1])
    plt.plot(x,y, color = color,
                             alpha = alpha,
                             linewidth = linewidth,
                             linestyle = linestyle)
    for t in templates:
        if t.idxs[0]>zoom_range[0] and t.idxs[1]<zoom_range[1]:
            x = range(t.idxs[0], t.idxs[1])
            y = t.template
            plt.plot(x,y, color = color, linewidth = 2)
            if end:
                plt.scatter(x[0], y[0], color = 'r')
                plt.scatter(x[-1], y[-1], color = 'r')
    if save == True:
        plt.savefig(name)
        

def pattern_centers_plot(centers:np.ndarray, names:list)->None:
    """
    Plot centers of clusters
    IN:
        centers - list of cluster centers
        names - indexes of cluster centers
        len(centers) == len(names)
    """
    i = 1
    w,h = optimal_size(len(centers))
    if w*h < len(centers):
        h+=1
    plt.figure(figsize=(w*1.5, h*1.5))
    for name, center in zip(names, centers):
        try:
            plt.subplot( h,w, i)
            plt.title(name)
            plt.plot(center)
    #         plt.axis('off')
            i+=1
        except:
            pass
    plt.tight_layout()
    