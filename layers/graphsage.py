import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class SAGELayer(nn.Module):
    
    def __init__(self, in_features, out_features, bias=False):
        super(SAGELayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(self.in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)


    def forward(self, input_, adj, norm=None):
        
        neighbor = torch.spmm(adj, input_)
        ft_input = torch.mm(input_, self.weight)
        ft_neighbor = torch.mm(neighbor, self.weight)

        output = torch.cat([ft_input, ft_neighbor], dim=1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
