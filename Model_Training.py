import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import models

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
def get_network_data(network):
    # read in edge list
    assert network in ["friend", "mention", "comment"]
    edge_list = pd.read_csv(directory + "/" + network + "_edges.csv")

    # convert data to appropriate form and generate an undirected PyG graph
    edge_index = torch.from_numpy(edge_list.to_numpy())
    net = Data(x=torch_features, edge_index=edge_index.t().contiguous(), y=torch_labels)

    # get train, val, test masks
    train_val_test_splitter = RandomNodeSplit(num_val=.2, num_test=.2)
    net = train_val_test_splitter(net)

    return net
    #
data = get_network_data("friend").to(device)
D_in = data.num_node_features

# Define Parameters
# H is hidden dimension; L is number of layers; lr is learning rate; wd is weight decay
H_list = [32]
L_list = [4]
lr_list = [1e-2,9e-1]
wd_list = [5e-4,1e-1]
num_epochs = 500
model_class = models.MultiLayerGCNNet

# constructs & trains model, then evaluates & returns MSE on eval_mask
def run(H,L,wd,lr,eval_mask):
    # construct model & optimizer
    model = model_class(D_in,H,L).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # model training
    model.train()
    for t in range(num_epochs):
        y_pred = model(data)  # forward pass
        loss = F.mse_loss(y_pred[data.train_mask], data.y[data.train_mask])  # compute loss
        if t % 100 == 99:
            #print(t + 1, loss.item())
            pass

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # get MSE values
    model.eval()
    mask = next(data(eval_mask))[1]
    pred = model(data)
    MSE = (pred[mask] - data.y[mask]).square().mean()
    return MSE.item()

# find best hyperparameter set
val_MSEs = []
index = 0
for H in H_list:
    for L in L_list:
        for wd in wd_list:
            for lr in lr_list:
                val_MSE = run(H, L, wd, lr, "val_mask")
                print('index {}: (H {}, layers {}, regularization {}, learning rate {}) = '.format(index,H,L,wd,lr), val_MSE)
                val_MSEs.append(val_MSE)
                index += 1
max_val = max(val_MSEs)
max_val_index = val_MSEs.index(max_val)
print('Best validation:', max_val)
print('Best index:', max_val_index)

# get best hyperparameters (by validation MSE)
best_H = H_list[max_val_index // (len(L_list)*len(wd_list)*len(lr_list))]
best_L = L_list[(max_val_index % (len(L_list)*len(wd_list)*len(lr_list))) // (len(wd_list)*len(lr_list))]
best_wd = wd_list[(max_val_index % (len(wd_list)*len(lr_list))) // len(lr_list)]
best_lr = lr_list[max_val_index % len(lr_list)]

# get test MSE
test_MSE = run(best_H,best_L,best_wd,best_lr,"test_mask")
print("Test MSE:",test_MSE)