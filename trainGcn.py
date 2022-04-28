import numpy as np
import scipy.sparse as sp
from utils import preprocess_graph, sparse_to_tuple
from models.gcnModel import GCNModelVAE
import torch
from tqdm import tqdm,trange
import torch.nn as nn
import torch.optim as optim
def sparse_to_dense(sparse):
    count = 0
    metrics = np.zeros(sparse[2])
    for index in sparse[0]:
        row = int(index[0])
        col = int(index[1])
        metrics[row][col] = sparse[1][count]
        count = count + 1
    return metrics


class VAELoss(nn.Module):
    def __init__(self,num_nodes,model,norm=1):
        super(VAELoss, self).__init__()
        self.norm = norm
        #self.preds = preds
        #self.labels = labels
        self.num_nodes = num_nodes
        self.model = model
        self.CEloss = nn.BCEWithLogitsLoss()
        # self.cost = nn.CrossEntropyLoss()
    def forward(self, preds,labels):

        cost = self.norm * torch.mean(self.CEloss(preds,labels))
        log_lik = cost
        kl = (0.5 / self.num_nodes) * torch.mean(torch.sum(1 + 2 * self.model.z_log_std_value - torch.pow(self.model.z_mean_value,2) - torch.pow(torch.exp(self.model.z_log_std_value),2), 1))
        cost -= kl

        return cost

def train_gcn(features, adj_train, args, graph_type,device):
    model_str = args.model

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj_train
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj = adj_train
    adj_norm = preprocess_graph(adj)
    adj_norm_dense = torch.Tensor(sparse_to_dense(adj_norm)).cuda(device)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    adj_label_dense = torch.Tensor(sparse_to_dense(adj_label)).cuda(device)
    feature = torch.Tensor(features.toarray()).cuda(device)
    # adj = torch.Tensor(adj_train.toarray()).cuda(device)

    # Some preprocessing

    num_nodes = adj.shape[0]
    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    model = None
    if graph_type == "sequence_similarity":
        epochs = args.epochs_simi
    else:
        epochs = args.epochs_ppi
    if model_str == 'gcn_vae':
        model = GCNModelVAE(num_features=num_features, num_nodes=num_nodes, features_nonzero=features_nonzero, hidden1=args.hidden1, hidden2=args.hidden2,device=device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    vaeloss = VAELoss(adj.shape[0],model)
    for e in range(epochs):
        model.train()
        optimizer.zero_grad()
        emd,output = model(adj_norm_dense,feature)
        loss = vaeloss(torch.reshape(output,[1,-1]),torch.reshape(adj_label_dense,[1,-1]))
        loss.backward()
        optimizer.step()

        print("Epoch:", '%04d' % (e+1), "train_loss=", "{:.5f}".format(loss))
    print("Optimization Finished!")

    return emd
