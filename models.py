### UTILITY FILE WITH ALL MODELS ###
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

# Multi Layer GCN regression model
class MultiLayerGCNNet(torch.nn.Module):
    def __init__(self, D_in, H, L):
        super(MultiLayerGCNNet, self).__init__()
        self.L = L
        self.conv1 = GCNConv(D_in, H)  # first
        self.model = nn.ModuleList([self.conv1])  # adding first
        self.model.extend([GCNConv(H, H) for _ in range(self.L - 2)])  # adding others
        self.convL = GCNConv(H, 1)  # last
        self.model.append(self.convL)  # adding last

    def forward(self, data):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x, edge_index = data.x, data.edge_index

        h = F.relu(self.model[0](x, edge_index))  # relu applied
        for i in range(1, self.L - 1):
            h = F.relu(self.model[i](h, edge_index))  # relu applied
        return self.model[-1](h, edge_index)  # no relu

class GATNet(torch.nn.Module):
    def __init__(self, D_in, H, L):
 
        super(GATNet, self).__init__()
        self.conv1 = GATConv(D_in, H)
        self.conv2 = GATConv(H, 1)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
