import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import models
import matplotlib.pyplot as plt
import numpy as np

### PARAMETERS TO SET ###
# chosen network
network_name = "comment"
is_undirected = False

# Define Model Hyperparameters
# H is hidden dimension; L is number of layers; lr is learning rate; wd is weight decay
H = 16
L = 2
lr = .01
wd = .0005
num_epochs = 500
model_class = models.MultiLayerGeneralConvNet

### DATA SETUP ###
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# get feature/label data
directory = "ProcessedData"
features = pd.read_csv(directory + "/features.csv")
labels = pd.read_csv(directory + "/labels.csv")
torch_features = torch.from_numpy(features.iloc[:, 1:].to_numpy()).float()
torch_labels = torch.from_numpy(labels.iloc[:, [1]].to_numpy()).float()

# Constructs PyG Data structure for specified network
# network = "friend", "mention", or "comment"
# returns PyG Data structure
def get_network_data(network,is_undirected):
    # read in edge list
    assert network in ["friend", "mention", "comment"]
    if(is_undirected):
        path = directory + "/" + network + "_edges_undirected.csv"
    else:
        path = directory + "/" + network + "_edges_directed.csv"
    edge_list = pd.read_csv(path)

    # convert data to appropriate form and generate an undirected PyG graph
    edge_index = torch.from_numpy(edge_list.to_numpy())
    net = Data(x=torch_features, edge_index=edge_index.t().contiguous(), y=torch_labels)

    # get train, val, test masks
    train_val_test_splitter = RandomNodeSplit(num_val=0, num_test=.2)
    net = train_val_test_splitter(net)

    return net
    #
data = get_network_data(network_name,is_undirected).to(device)
D_in = data.num_node_features

### MODEL TRAINING/EVALUATION ###
# constructs & trains model, then evaluates & returns MSEs validation & test sets
def run(H,L,wd,lr):
    # construct model & optimizer
    model = model_class(D_in,H,L).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # model training
    model.train()
    train_losses = []
    test_losses = []
    for t in range(num_epochs):
        y_pred = model(data)  # forward pass
        loss = F.mse_loss(y_pred[data.train_mask], data.y[data.train_mask])  # compute loss
        if t % 100 == 99:
            print("Train rMSE " + str(t + 1) + ":", loss.sqrt().item())
            pass

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store losses
        train_losses.append(loss.sqrt().item())
        with torch.no_grad():
            for _, mask in data('test_mask'):
                pred = model(data)
                rMSE = (pred[mask] - data.y[mask]).square().mean().sqrt().item()
                test_losses.append(rMSE)

    return train_losses, test_losses

# find best hyperparameter set
train_losses, test_losses = run(H, L, wd, lr)

x_axis = np.arange(num_epochs)
plt.plot(x_axis,np.array(train_losses),label="Train Loss")
plt.plot(x_axis,np.array(test_losses),label="Test Loss")
plt.legend()
plt.title("RMSE Loss Plot")
plt.xlabel("epoch")
plt.ylabel("RMSE")
plt.show()
