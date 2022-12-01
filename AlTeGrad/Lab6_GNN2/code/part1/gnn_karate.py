"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from models import GNN
from utils import sparse_to_torch_sparse


# Initialize device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Hyperparameters
epochs = 200
n_hidden = 16
learning_rate = 0.01
dropout_rate = 0.1

# Loads the karate network
G = nx.read_weighted_edgelist('AlTeGrad/Lab6_GNN2/code/data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print('Number of nodes:', G.number_of_nodes())
print('Number of edges:', G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('AlTeGrad/Lab6_GNN2/code/data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)
n_class = 2

############## Task 3
adj = nx.adjacency_matrix(G) # your code here #
features = np.random.randn(n, 4) # your code here #

# Yields indices to split data into training and test sets
idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

# Transforms the numpy matrices/vectors to torch tensors
features = torch.FloatTensor(features).to(device)
y = torch.LongTensor(y).to(device)
adj = sparse_to_torch_sparse(adj).to(device)
idx_train = torch.LongTensor(idx_train).to(device)
idx_test = torch.LongTensor(idx_test).to(device)

# Creates the model and specifies the optimizer
model = GNN(features.shape[1], n_hidden, n_class, dropout_rate).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train model
for epoch in range(epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output, _ = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], y[idx_train])
    acc_train = accuracy_score(torch.argmax(output[idx_train], dim=1).detach().cpu().numpy(), y[idx_train].cpu().numpy())
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:03d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train),
          'time: {:.4f}s'.format(time.time() - t))

print("Optimization Finished!")

# Testing
model.eval()
output, alpha = model(features, adj)
loss_test = F.nll_loss(output[idx_test], y[idx_test])
acc_test = accuracy_score(torch.argmax(output[idx_test], dim=1).detach().cpu().numpy(), y[idx_test].cpu().numpy())
print("Test set results:",
      "loss= {:.4f}".format(loss_test.item()),
      "accuracy= {:.4f}".format(acc_test))


############## Task 4
alpha = alpha.detach().cpu().numpy() # your code here #

# Dictionary that maps indices of nodes to nodes
idx_to_node = dict()
for i,node in enumerate(G.nodes()):
    idx_to_node[i] = node

# Creates a directed karate network
G_directed = G.to_directed()

# Retrieves nonzero indices of the adjacency matrix
indices = adj.coalesce().indices().detach().cpu().numpy()

# Annotates edges with the learned attention weights 
for i in range(indices.shape[1]):
    G_directed[idx_to_node[indices[0,i]]][idx_to_node[indices[1,i]]]['weight'] = alpha[i]

weights = [G_directed[u][v]['weight'] for u,v in G_directed.edges()]

# Visualizes attention weights
plt.figure(1,figsize=(12,12))
pos = nx.spring_layout(G_directed)
arc_rad = 0.25
nx.draw(G_directed, width=weights, connectionstyle=f'arc3, rad = {arc_rad}')
plt.show()