# Reduced parameter version of Relation Module

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class Relationv2(nn.Module):
    def __init__(self, in_features, out_features, ablation=0):
        super(Relationv2, self).__init__()
        
        self.gamma1_1 = nn.Linear(in_features, out_features, bias=False)
        self.gamma1_2 = nn.Linear(out_features, in_features, bias=False)

        self.gamma2_1 = nn.Linear(in_features, out_features, bias=False)
        self.gamma2_2 = nn.Linear(out_features, in_features, bias=False)

        self.beta1_1 = nn.Linear(in_features, out_features, bias=False)
        self.beta1_2 = nn.Linear(out_features, in_features, bias=False)

        self.beta2_1 = nn.Linear(in_features, out_features, bias=False)
        self.beta2_2 = nn.Linear(out_features, in_features, bias=False)

        self.r = Parameter(torch.FloatTensor(1, in_features))

        self.ablation = ablation
        self.elu = nn.ELU()
        self.lrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.reset_parameter()
        
        
    def weight_init(self, m):
        return

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.r.size(1))
        self.r.data.uniform_(-stdv, stdv)
        

    def forward(self, ft, neighbor):

        if self.ablation == 3:
            self.m = ft + self.r - neighbor
        else:

            gamma1 = self.gamma1_2(self.gamma1_1(ft))
            gamma2 = self.gamma2_2(self.gamma2_1(neighbor))
            gamma = self.lrelu(gamma1 + gamma2) + 1.0 

            beta1 = self.beta1_2(self.beta1_1(ft)) 
            beta2 = self.beta2_2(self.beta2_1(neighbor))
            beta = self.lrelu(beta1 + beta2) 

            self.r_v = gamma * self.r + beta
            self.m = ft + self.r_v - neighbor
            
        return F.normalize(self.m)