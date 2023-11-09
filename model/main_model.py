# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 14:36
# @Author  : Yanjun Hao
# @Site    : 
# @File    : main_model.py
# @Software: PyCharm 
# @Comment :


import numpy as np
from math import sqrt
from torch import nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features).type(torch.float32))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features).type(torch.float32))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # print(input.shape, adj.shape)
        support = torch.matmul(x, self.weight.type(torch.float32))
        output = torch.bmm(adj.unsqueeze(0).expand(support.size(0), *adj.size()), support)
        if self.bias is not None:
            return output + self.bias.type(torch.float32)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class Model(nn.Module):
    def __init__(self, time_lag, pre_len, station_num, device):
        super().__init__()
        self.time_lag = time_lag
        self.pre_len = pre_len
        self.station_num = station_num
        self.device = device
        self.GCN1 = GraphConvolution(in_features=self.time_lag, out_features=self.time_lag).to(self.device)
        self.GCN2 = GraphConvolution(in_features=self.time_lag, out_features=self.time_lag).to(self.device)
        self.linear1 = nn.Linear(in_features=self.time_lag * self.station_num, out_features=1024).to(self.device)
        self.linear2 = nn.Linear(in_features=1024, out_features=512).to(self.device)
        self.linear3 = nn.Linear(in_features=512, out_features=self.station_num * self.pre_len).to(self.device)

    def forward(self, speed, adj):
        speed = speed.to(self.device)  # [32,156,10]
        adj = adj.to(self.device)
        speed = self.GCN1(x=speed, adj=adj)  # (32, 156, 10)
        output = self.GCN2(x=speed, adj=adj)  # [32, 156, 10]
        output = output.reshape(output.size()[0], -1)  # (32, 156*10)
        output = F.relu(self.linear1(output))  # (32, 1024)
        output = F.relu(self.linear2(output))  # (32, 512)
        output = self.linear3(output)  # (32, 276*pre_len)
        output = output.reshape(output.size()[0], self.station_num, self.pre_len)  # ( 64, 276, pre_len)
        return output
