"""
Graph Mining - ALTEGRAD - Nov 2022
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# networkx works not very well on large scale. 

############## Task 1

##################
# your code here #
# This is a directed graph 
G = nx.read_edgelist('AlTeGrad/Lab4_GraphMining/code/datasets/CA-HepTh.txt', comments='#', delimiter='\t')
print(f"The graph has {G.number_of_nodes()} nodes.")
print(f"The graph has {G.number_of_edges()} edges.")

##################



############## Task 2

##################
# your code here #

print(f"The graph has {nx.number_connected_components(G)} connected components.")
largest_cc = max(nx.connected_components(G), key=len)

subG = G.subgraph(largest_cc)

print(f"The largest connected components in graph has {subG.number_of_edges()} edges.")
##################



############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
print(f"The minimum of degree in the graph is {np.min(degree_sequence)}")
print(f"The maximum of degree in the graph is {np.max(degree_sequence)}")
print(f"The median of degree in the graph is {np.median(degree_sequence)}")
print(f"The mean of degree in the graph is {np.mean(degree_sequence)}")

##################



############## Task 4

##################
# your code here #
plt.subplot(211)
hist, bins, _ = plt.hist(degree_sequence, bins=10)

# histogram on log scale. 
# Use non-equal bin sizes, such that they look equal on log scale.
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.subplot(212)
plt.hist(degree_sequence, bins=logbins)
plt.xscale('log')
plt.show()
##################




############## Task 5

##################
# your code here #
print(f"The global clustering coefficent of graph is {nx.transitivity(G)}.")
##################
