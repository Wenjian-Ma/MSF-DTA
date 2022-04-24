import argparse
from utils import load_data
from trainGcn import train_gcn
import numpy as np
import pandas as pd
import os

import torch
import json
def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))

def train(args):
    # load feature dataframe
    print("loading features...")
    uniprot = pd.read_pickle("./data/networks/features.pkl")
    device = torch.device('cuda:1')

    embeddings_list = []
    for graph in args.graphs:
        print("#############################")
        print("Training",graph)
        adj, features = load_data(graph, uniprot, args)
        embeddings = train_gcn(features, adj, args, graph,device)
        embeddings_list.append(embeddings.cpu().detach().numpy())
    embeddings = np.hstack(embeddings_list)
    np.save('/home/sgzhang/perl5/MSF-DTA/data/embeddings/embeddings.npy',embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # global parameters
    parser.add_argument('--ppi_attributes', type=int, default=6, help="types of attributes used by ppi.")
    parser.add_argument('--simi_attributes', type=int, default=6, help="types of attributes used by simi.")
    parser.add_argument('--graphs', type=lambda s: [item for item in s.split(",")],
                        default=['ppi', 'sequence_similarity'], help="lists of graphs to use.")
    parser.add_argument('--thr_combined', type=float, default=0.3, help="threshold for combiend ppi network.")#0.4
    parser.add_argument('--thr_evalue', type=float, default=1e-4, help="threshold for similarity network.")#1e-4
    parser.add_argument('--supervised', type=str, default="nn", help="neural networks or svm")
    parser.add_argument('--only_gcn', type=int, default=0, help="0 for training all, 1 for only embeddings.")
    # parameters for traing GCN
    parser.add_argument('--lr', type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument('--epochs_ppi', type=int, default=180, help="Number of epochs to train ppi.")#150
    parser.add_argument('--epochs_simi', type=int, default=130, help="Number of epochs to train similarity network.")#130
    parser.add_argument('--hidden1', type=int, default=800, help="Number of units in hidden layer 1.")
    parser.add_argument('--hidden2', type=int, default=400, help="Number of units in hidden layer 2.")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight for L2 loss on embedding matrix.")
    parser.add_argument('--dropout', type=float, default=0, help="Dropout rate (1 - keep probability).")
    parser.add_argument('--model', type=str, default="gcn_vae", help="Model string.")

    args = parser.parse_args()
    print(args)
    train(args)