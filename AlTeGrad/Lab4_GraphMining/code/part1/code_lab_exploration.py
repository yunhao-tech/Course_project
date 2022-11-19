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
G = nx.read_edgelist('datasets/CA-HepTh.txt', comments='#', delimiter='\t')
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
##################



############## Task 4

##################
# your code here #
##################




############## Task 5

##################
# your code here #
print(f"The global clustering coefficent of graph is {nx.transitivity(G)}.")
##################
