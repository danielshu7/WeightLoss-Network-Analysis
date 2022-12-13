### UTILITY FILE WITH ALL MODELS ###
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SuperGATConv, GeneralConv 

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

# Multi Layer GATConv regression model
class GATNet(torch.nn.Module):
    def __init__(self, D_in, H, L):
        super(GATNet, self).__init__()
        self.L = L
        self.conv1 = GATConv(D_in, H)  # first
        self.model = nn.ModuleList([self.conv1])  # adding first
        self.model.extend([GATConv(H, H) for _ in range(self.L - 2)])  # adding others
        self.convL = GATConv(H, 1)  # last
        self.model.append(self.convL)  # adding last

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = F.relu(self.model[0](x, edge_index))  # relu applied
        for i in range(1, self.L - 1):
            h = F.relu(self.model[i](h, edge_index))  # relu applied
        return self.model[-1](h, edge_index)  # no relu
    
# Multi Layer SuperGATConv regression model
class MultiLayerSuperGATConvNet(torch.nn.Module):
    def __init__(self, D_in, H, L):
        super(MultiLayerSuperGATConvNet, self).__init__()
        self.L = L
        self.conv1 = SuperGATConv(D_in, H)  # first
        self.model = nn.ModuleList([self.conv1])  # adding first
        self.model.extend([SuperGATConv(H, H) for _ in range(self.L - 2)])  # adding others
        self.convL = SuperGATConv(H, 1)  # last
        self.model.append(self.convL)  # adding last

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = F.relu(self.model[0](x, edge_index))  # relu applied
        for i in range(1, self.L - 1):
            h = F.relu(self.model[i](h, edge_index))  # relu applied
        return self.model[-1](h, edge_index)  # no relu
    
# Multi Layer GeneralConv regression model
class MultiLayerGeneralConvNet(torch.nn.Module):
    def __init__(self, D_in, H, L):
        super(MultiLayerGeneralConvNet, self).__init__()
        self.L = L
        # Attention parameter = true because MSE is much higher without it for all hyperparameters
        self.conv1 = GeneralConv(D_in, H, attention=True)  # first
        self.model = nn.ModuleList([self.conv1])  # adding first
        self.model.extend([GeneralConv(H, H, attention=True) for _ in range(self.L - 2)])  # adding others
        self.convL = GeneralConv(H, 1, attention=True)  # last
        self.model.append(self.convL)  # adding last

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = F.relu(self.model[0](x, edge_index))  # relu applied
        for i in range(1, self.L - 1):
            h = F.relu(self.model[i](h, edge_index))  # relu applied
        return self.model[-1](h, edge_index)  # no relu

class Regression(torch.nn.Module):
    def __init__(self, D_in, H, L):
        super(Regression, self).__init__()
        self.L = L
        self.hid1 = torch.nn.Linear(D_in, H) 
        self.hid2 = torch.nn.Linear(H, H)
        self.oupt = torch.nn.Linear(H, 1)

    def forward(self, data):
        x = data.x
        z = torch.relu(self.hid1(x))
        z = torch.relu(self.hid2(z))
        z = self.oupt(z)  
        return z