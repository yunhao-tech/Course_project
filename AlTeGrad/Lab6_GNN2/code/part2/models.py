"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, hidden_dim_3, n_class, device):
        super(GNN, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, hidden_dim_3)
        self.fc4 = nn.Linear(hidden_dim_3, n_class)
        self.relu = nn.ReLU()

    def forward(self, x_in, adj, idx):
        
        ############## Task 6
    
        ##################
        # your code here #
        adj = adj + np.eye(adj.shape[0])
        x = self.relu(torch.mm(adj, self.fc1(x_in))) # Relu(A @ X @ W0), 
        
        x = self.relu(torch.mm(adj, self.fc2(x))) # Relu(A @ X @ W1), 
        ##################
        
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1), dtype=x.dtype).to(self.device)
        out = out.scatter_add_(0, idx, x) 
        
        ##################
        # your code here #
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        ##################

        return F.log_softmax(out, dim=1)
