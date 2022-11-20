"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans
from collections import defaultdict


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
G = nx.read_edgelist('AlTeGrad/Lab4_GraphMining/code/datasets/CA-HepTh.txt', comments='#', delimiter='\t')
largest_cc = max(nx.connected_components(G), key=len)
subG = G.subgraph(largest_cc)
clusterings = spectral_clustering(G, 50)

##################



############## Task 8
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    res = defaultdict(list) # cluster number -> list of nodes
    for key, val in sorted(clustering.items()):
        res[val].append(key)
    l_c = [G.subgraph(res[key]).number_of_edges() for key, _ in res.items()]
    d_c = [sum([G.degree(node) for node in value]) for _, value in res.items()]
    m = G.number_of_edges() # total number of edges
    l_c_m = [ele / m for ele in l_c]
    d_c_m = [(ele / (2*m))**2 for ele in d_c]
    assert(len(l_c_m) == len(d_c_m))
    modularity = sum([ele1 - ele2 for ele1, ele2 in zip(l_c_m, d_c_m)])
    ##################
    
    
    return modularity



############## Task 9

##################
# your code here #
print(f"The modularity of result of pectral Clustering is {modularity(G, clusterings)}")

random_clusterings = {}
for node in G.nodes():
    random_clusterings[node] = randint(0, 49)
print(f"The modularity of random clustering is {modularity(G, random_clusterings)}")

##################







