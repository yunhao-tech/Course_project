"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """Simple GNN model"""
    def __init__(self, n_feat, n_hidden_1, n_hidden_2, n_class, dropout):
        super(GNN, self).__init__()

        self.fc1 = nn.Linear(n_feat, n_hidden_1)
        self.fc2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.fc3 = nn.Linear(n_hidden_2, n_class)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj):
        ############## Tasks 10 and 13
        
        ##################
        # your code here #
        # self.fc1(x_in) = X @ W0
        # adj = A
        z0 = self.relu(torch.mm(adj, self.fc1(x_in))) # Relu(A @ X @ W0), 
        z0 = self.dropout(z0)
        
        z1 = self.relu(torch.mm(adj, self.fc2(z0))) # Relu(A @ X @ W1), 
        z1 = self.dropout(z1)
        
        x = self.fc3(z1)
        ##################


        return F.log_softmax(x, dim=1)