"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 6
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    # your code here #
    A = nx.adjacency_matrix(G)
    D_inv = diags([1/G.degree(node) for node in G.nodes()])
    
    Lrw = eye(G.number_of_nodes()) - D_inv @ A
    
    evals, evects = eigs(Lrw, k=k, which='SR')
    evects = np.real(evects)
    
    kmeans = KMeans(n_clusters=k).fit(evects)
    
    clustering = {}
    
    for i, node in enumerate(G.nodes()):
        clustering[node] = kmeans.labels_[i]
        
    ##################
    

    
    return clustering





############## Task 7

##################
# your code here #
##################





############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    
    ##################
    

    
    
    return modularity



############## Task 9

##################
# your code here #
##################







