import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
from layers import *

class TransGCN_SP(nn.Module):
    def __init__(self, nfeat, nhid, g_sigma, ver, ablation=0):
        super(TransGCN_SP, self).__init__()
        
        if ver == 1:
            self.r = Relation(nfeat, ablation)
        else:
            self.r = Relationv2(nfeat,nhid, ablation=ablation)

        self.g = Generator(nfeat, g_sigma, ablation)
        self.gc = GraphConv(nfeat, nhid)
        self.ablation = ablation


    def forward(self, x, adj, adj_self, head, norm):
        
        #norm = sp.sum(adj, dim=1).to_dense().view(-1,1)
        neighbor = sp.mm(adj, x)
        m = self.r(x, neighbor)

        if head or self.ablation == 2:
            #norm = sp.sum(adj_self, dim=1).to_dense().view(-1,1)
            h_k = self.gc(x, adj_self, norm=norm)
        else:
            if self.ablation == 1:
                h_s = self.g(m)
            else:
                h_s = m
            
            h_s = torch.mm(h_s, self.gc.weight)
            h_k = self.gc(x, adj_self)
            h_k = (h_k + h_s) / (norm + 1)
        
        return h_k, m 


class TransGAT_SP(nn.Module):
    def __init__(self, nfeat, nhid, g_sigma, device, ver, ablation=0, nheads=3, dropout=0.5, concat=True):
        super(TransGAT_SP, self).__init__()
        
        self.ablation = ablation
        if ver == 1:
            self.r = Relation(nfeat, ablation=ablation)
        else:
            self.r = Relationv2(nfeat,nhid, ablation=ablation)

        self.g = Generator(nfeat, g_sigma, ablation)        
        self.gat = [SpGraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=0.2, concat=concat) for _ in range(nheads)]
        for i, attention in enumerate(self.gat):
            self.add_module('attention_{}'.format(i), attention)


    def forward(self, x, adj, adj_self, head, norm):
        
        neighbor = sp.mm(adj, x)
        m = self.r(x, neighbor)
       
        if head or self.ablation == 2:
            h_k = torch.cat([att(x, adj_self) for att in self.gat], dim=1)
        else:
            h_k = torch.cat([att(x, adj_self, mi=m) for att in self.gat], dim=1)
        
        return h_k, m




# latent relation GCN
class TailGCN_SP(nn.Module):
    def __init__(self, nfeat, nclass, params, device, ver=1, ablation=0):
        super(TailGCN_SP, self).__init__()

        self.device = device
        self.nhid = params.hidden
        self.dropout = params.dropout
        self.ablation = ablation

        #self.rel1 = TransGCN_SP(nfeat, self.nhid, g_sigma=params.g_sigma, ver=ver)    
        if ver == 1:
            self.r1 = Relation(nfeat, ablation=ablation)
        else:
            self.r1 = Relationv2(nfeat, self.nhid, ablation=ablation)
        self.g1 = Generator(nfeat, params.g_sigma,ablation).to(device)
        
        self.gc1 = GraphConv(nfeat, self.nhid).to(device)
        self.rel2 = TransGCN_SP(self.nhid, nclass, g_sigma=params.g_sigma, ver=ver, ablation=ablation).to(device)            


    def forward(self, x, adj, head, adj_self=None, norm=None):
        
        #rewrite rel1
        neighbor = sp.mm(adj, x)
        m1 = self.r1(x, neighbor)

        x = x.to(self.device)
        m1 = m1.to(self.device)
        adj = adj.to(self.device)
        adj_self = adj_self.to(self.device)
        norm = norm.to(self.device)

        if head or self.ablation == 2:
            x1 = self.gc1(x, adj_self, norm=norm)
        else:
            if self.ablation == 1:
                h_s = self.g1(m1)
            else:
                h_s = m1
            
            h_s = torch.mm(h_s, self.gc1.weight)
            h_k = self.gc1(x, adj_self)
            x1 = (h_k + h_s) / (norm + 1)
    
        x1 = F.elu(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2, m2 = self.rel2(x1, adj, adj_self, head, norm)
        norm_m1 = torch.norm(m1, dim=1)
        norm_m2 = torch.norm(m2, dim=1)
        
        return x2, norm_m1, norm_m2 #, head_prob, tail_prob



class TailGAT_SP(nn.Module):
    def __init__(self, nfeat, nclass, params, device, ver=1, ablation=0):
        super(TailGAT_SP, self).__init__()

        self.device = device
        self.nhid = params.hidden
        self.dropout = params.dropout
        self.ablation = ablation
        
        nheads = 3
        nhid = 8

        if ver == 1:
            self.r1 = Relation(nfeat, ablation=ablation)
        else:
            self.r1 = Relationv2(nfeat, self.nhid, ablation=ablation)

        self.gat1 = [SpGraphAttentionLayer(nfeat, nhid, dropout=self.dropout, alpha=0.2, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.gat1):
            self.add_module('attention_{}'.format(i), attention)
        self.rel2 = TransGAT_SP(nhid * nheads, nclass, g_sigma=params.g_sigma, device=device, ver=ver, ablation=ablation, nheads=1).to(device)
    

    def forward(self, x, adj, head, adj_self=None, norm=None):
        #rewrite rel1
        neighbor = sp.mm(adj, x)
        m1 = self.r1(x, neighbor)

        if head or self.ablation == 2:
            x1 = torch.cat([att(x, adj_self) for att in self.gat1], dim=1)
        else:
            x1 = torch.cat([att(x, adj_self, mi=m1) for att in self.gat1], dim=1)
        
        x1 = x1.to(self.device)
        m1 = m1.to(self.device)
        adj = adj.to(self.device)
        adj_self = adj_self.to(self.device)

        x1 = F.elu(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)

        x2, m2 = self.rel2(x1, adj, adj_self, head, norm)
        norm_m1 = torch.norm(m1, dim=1)
        norm_m2 = torch.norm(m2, dim=1)
        
        return x2, norm_m1, norm_m2



class TailSAGE_SP(nn.Module):
    def __init__(self, nfeat, nclass, params, device, ver=1, ablation=0):
        super(TailSAGE_SP, self).__init__()

        self.device = device
        self.nhid = params.hidden
        self.dropout = params.dropout
        self.ablation = ablation

        if ver == 1:
            self.r1 = Relation(nfeat, ablation=ablation)
            self.r2 = Relation(self.nhid*2, ablation=ablation).to(device)
        else:
            self.r1 = Relationv2(nfeat, self.nhid, ablation=ablation)
            self.r2 = Relationv2(self.nhid*2, nclass, ablation=ablation).to(device)

        self.w1 = nn.Linear(nfeat, self.nhid, bias=False).to(device)
        self.w2 = nn.Linear(self.nhid * 2, nclass, bias=False).to(device)
        self.fc = nn.Linear(nclass * 2, nclass, bias=True).to(device)

    def forward(self, x, adj, head, adj_self=None, norm=None):
        
        # rel1
        neighbor = sp.mm(adj, x)
        m1 = self.r1(x, neighbor)

        x = x.to(self.device)
        neighbor = neighbor.to(self.device)
        m1 = m1.to(self.device)
        adj = adj.to(self.device)
        norm = norm.to(self.device)

        if head:
            ft_input = self.w1(x)
            ft_neighbor = self.w1(neighbor)
            x1 = torch.cat([ft_input, ft_neighbor], dim=1)

        else:
            neighbor = neighbor + m1 / (norm+1)
            ft_input = self.w1(x)
            ft_neighbor = self.w1(neighbor)
            x1 = torch.cat([ft_input, ft_neighbor], dim=1)
    
        x1 = F.elu(x1)
        x1 = F.normalize(x1)
        x1 = F.dropout(x1, self.dropout, training=self.training)

        # rel2
        neighbor1 = sp.mm(adj, x1)
        m2 = self.r2(x1, neighbor1)
        
        if head:
            ft_input1 = self.w2(x1)
            ft_neighbor1 = self.w2(neighbor1)
            x2 = torch.cat([ft_input1, ft_neighbor1], dim=1)

        else:
            neighbor1 = neighbor1 + m2 / (norm+1)
            ft_input1 = self.w2(x1)
            ft_neighbor1 = self.w2(neighbor1)
            x2 = torch.cat([ft_input1, ft_neighbor1], dim=1)
    
        x2 = F.elu(x2)
        x2 = F.normalize(x2)

        x2 = self.fc(x2)
        norm_m1 = torch.norm(m1, dim=1)
        norm_m2 = torch.norm(m2, dim=1)
        
        return x2, norm_m1, norm_m2
