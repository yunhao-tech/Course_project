"""
Learning on Sets / Learning with Proteins - ALTEGRAD - Dec 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNN(nn.Module):
    """
    Simple message passing model that consists of 2 message passing layers
    and the sum aggregation function
    """
    def __init__(self, input_dim, hidden_dim, dropout, n_class):
        super(GNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, n_class)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_in, adj, idx):
        
        ############## Task 10
    
        ##################
        # your code here #
        x = self.relu(torch.mm(adj, self.fc1(x_in))) # Relu(A @ X @ W0), 
        x = self.dropout(x)

        x = torch.mm(adj, self.fc2(x)) # A @ X @ W1, 
        ##################
        
        # sum aggregator
        idx = idx.unsqueeze(1).repeat(1, x.size(1))
        out = torch.zeros(torch.max(idx)+1, x.size(1)).to(x_in.device)
        out = out.scatter_add_(0, idx, x)
        
        ##################
        # your code here #
        out = self.bn(out)

        out = self.relu(self.fc3(out))
        out = self.dropout(out)
        out = self.fc4(out)
        ##################

        return F.log_softmax(out, dim=1)
