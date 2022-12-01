"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('AlTeGrad/Lab5_DeepWalk_GCN/code/data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('AlTeGrad/Lab5_DeepWalk_GCN/code/data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


############## Task 5
# Visualizes the karate network

##################
# your code here #
nx.draw_networkx(G, node_color=y)
##################


############## Task 6
# Extracts a set of random walks from the karate network and feeds them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim) # your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions


##################
# your code here #
clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy using Deepwalk embeddings is {accuracy_score(y_test, y_pred)}")
##################


############## Task 8
# Generates spectral embeddings

##################
# your code here #
k = 2
A = nx.adjacency_matrix(G)
D_inv = diags([1/G.degree(node) for node in G.nodes()])
Lrw = eye(G.number_of_nodes()) - D_inv @ A
evals, evecs = eigs(Lrw, k=k, which='SR')
spectral_embeddings = np.real(evecs)

X_train = spectral_embeddings[idx_train, :]
X_test = spectral_embeddings[idx_test, :]

clf = LogisticRegression(solver='lbfgs')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy using spectral embeddings is {accuracy_score(y_test, y_pred)}")
##################