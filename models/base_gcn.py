import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
from layers import GraphConv, GraphAttentionLayer, SpGraphAttentionLayer, SAGELayer


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):        
        x = F.relu(self.gc1(x, adj))        
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        #x = F.dropout(x, self.dropout, training=self.training)
        return x 



class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, alpha=0.2, nheads=3):
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)


    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class GraphSAGE(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclass, dropout=0.5, alpha=0.2, nheads=3):
        super(GraphSAGE, self).__init__()

        self.sage1 = SAGELayer(nfeat, nhid1)
        self.sage2 = SAGELayer(nhid1*2, nhid2)
        self.fc = nn.Linear(nhid2*2, nclass, bias=True)
        self.dropout = dropout

    def forward(self, x, adj):

        x = F.relu(self.sage1(x, adj))
        x = F.normalize(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.sage2(x, adj))
        x = F.normalize(x)

        x = self.fc(x)
        return x #F.log_softmax(x, dim=1)