import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, input_dim=78, output_dim=10, dropout=0.2, pretrain_flag=False):

        super(GINConvNet, self).__init__()
        self.flag = pretrain_flag
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        # convolution layers
        nn1 = Sequential(Linear(input_dim, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # combined layers
        self.out = nn.Linear(output_dim, output_dim)


    def forward(self, data, pretrain_flag=False):
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        # x = self.bn1(x)
        x = self.relu(self.conv2(x, edge_index))
        # x = self.bn2(x)
        x = self.relu(self.conv3(x, edge_index))
        # x = self.bn3(x)
        x = self.relu(self.conv4(x, edge_index))
        # x = self.bn4(x)
        x = self.relu(self.conv5(x, edge_index))
        # x = self.bn5(x)
        if self.flag == False:
            x = global_add_pool(x, batch=None)
        x = self.relu(self.fc1_xd(x))
        x = self.dropout(x)
        x = self.out(x)
        return x
