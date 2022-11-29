"""
Deep Learning on Graphs - ALTEGRAD - Nov 2022
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """GAT layer"""
    def __init__(self, n_feat, n_hidden, alpha=0.05):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(n_feat, n_hidden, bias=False)
        self.a = nn.Linear(2*n_hidden, 1)
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x, adj):
        
        ############## Task 1
    
        ##################
        # your code here #
        z = self.fc(x) # update the node representation!
        indices = adj.coalesce().indices() # a tensor of (2 * 2m) where m is number of edges
        h = torch.cat((z[indices[0,:], :], z[indices[1,:], :]), dim=1)
        h = self.leakyrelu(self.a(h))
        ##################
        # apply softmax
        h = torch.exp(h.squeeze())
        unique = torch.unique(indices[0,:])
        t = torch.zeros(unique.size(0), device=x.device)
        h_sum = t.scatter_add(0, indices[0,:], h)
        h_norm = torch.gather(h_sum, 0, indices[0,:])
        alpha = torch.div(h, h_norm)
        adj_att = torch.sparse.FloatTensor(indices, alpha, torch.Size([x.size(0), x.size(0)])).to(x.device)
        
        ##################
        # your code here #
        out = torch.sparse.mm(adj_att, z)
        ##################

        return out, alpha


class GNN(nn.Module):
    """GNN model"""
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GNN, self).__init__()
        self.mp1 = GATLayer(nfeat, nhid)
        self.mp2 = GATLayer(nhid, nhid)
        self.fc = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, adj):
        
        ############## Tasks 2 and 4
    
        ##################
        # your code here #
        z1, _ = self.mp1(x, adj)
        z1 = self.dropout(self.relu(z1))
        z2, alpha = self.mp2(z1, adj)
        z2 = self.relu(z2)
        out = self.fc(z2)
        ##################

        return F.log_softmax(out, dim=1), alpha