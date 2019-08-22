from sklearn.cluster import AffinityPropagation
import numpy as np
from scipy.spatial import distance

import sys
sys.path.append('../../ikmeans')
from ikmeans_error import IKMeansError 


def update_centers(centers):
    clustering = AffinityPropagation(0.9).fit(centers)
    labels =  clustering.labels_
    centers_new= []
    for i in range(max(labels)):
        centers_idx = centers[labels==i]
        centers_new.append(np.mean(centers_idx, axis = 0)) 
    if len(centers_new) == 0:
        raise IKMeansError("Reclustering produced 0 clusters. Try to" +
                           " increase the initial number of clusters or change the DWT level.")
    return np.asarray(centers_new)

def new_centers(centers):    
    centers = update_centers(centers)
    return np.append(centers, centers, axis = 1)

def nearest(patterns, template, k_nearest = 30):
    distations = list(map(lambda e: distance.euclidean(e.ca, template), patterns))
    idxs = list(range(0,len(patterns)))
    p_dist = list(zip(patterns, distations, idxs))
    return np.asarray(sorted(p_dist, key = lambda e:e[1])[:k_nearest])

def get_temp_info(temp):
    agrig_data = []
    orig_data = []
    
    for t in temp:
        agrig_data.append(t.ca)
        orig_data.append(t.template_norm)
    return agrig_data, orig_data