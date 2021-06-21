import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super(Discriminator, self).__init__()
        
        self.d = nn.Linear(in_features, in_features, bias=True)
        self.wd = nn.Linear(in_features, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    '''
    def weight_init(self, m):
        if isinstance(m, Parameter):
            torch.nn.init.xavier_uniform_(m.weight.data)

        if isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:                
                m.bias.data.uniform_(-stdv, stdv)
    '''

    def forward(self, ft):

        ft = F.elu(ft)
        ft = F.dropout(ft, 0.5, training=self.training)

        fc = F.elu(self.d(ft))
        prob = self.wd(fc)
        
        return self.sigmoid(prob)
