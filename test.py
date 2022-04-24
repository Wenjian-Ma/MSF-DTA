import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    # print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = [['davis', 'kiba'][int(sys.argv[1])]]
modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6"][int(sys.argv[3])]
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 1500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if (not os.path.isfile(processed_data_file_test)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root='data', dataset=dataset + '_test')

        # make data PyTorch mini-batch processing ready
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model_path = './data/processed/models/model_'+model_st+'_'+dataset+'.pkl'

        params_dict = torch.load(model_path)
        model = modeling().to(device)
        model.load_state_dict(params_dict)
        loss_fn = nn.MSELoss()
        print('predicting for test data')
        G, P = predicting(model, device, test_loader)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
        best_test_mse = ret[1]
        best_test_ci = ret[-1]
        print('best_test_mse,best_test_ci:', best_test_mse, best_test_ci, model_st, dataset)



