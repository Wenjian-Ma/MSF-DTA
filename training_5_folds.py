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


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    # print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()

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
    cuda_name = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6","cuda:7"][int(sys.argv[3])]
print('cuda_name:', cuda_name)

folds_split = int(sys.argv[4])

TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LR = 0.0005
NUM_EPOCHS = 1500

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# mark_split = [0,1,2,3,4]

MSE_score = []
CI_score = []

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        train_data = TestbedDataset(root='data', dataset=dataset + '_train')
        train_size = int(0.8 * len(train_data))
        valid_size = len(train_data) - train_size

        if folds_split == 0:
            valid_data,train_data = train_data[:valid_size],train_data[valid_size:]
        elif folds_split == 1:
            valid_data, train_data = train_data[valid_size:valid_size*2], train_data[:valid_size]+train_data[valid_size*2:]
        elif folds_split == 2:
            valid_data, train_data = train_data[valid_size*2:valid_size*3],train_data[:valid_size*2]+train_data[valid_size*3:]
        elif folds_split == 3:
            valid_data, train_data = train_data[valid_size*3:valid_size*4],train_data[:valid_size*3]+train_data[valid_size*4:]
        elif folds_split == 4:
            valid_data, train_data = train_data[valid_size*4:],train_data[:valid_size*4]
        #train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=False)


        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        best_mse = 1000
        best_test_mse = 1000
        best_test_ci = 0
        best_epoch = -1
        model_file_name = '/home/sgzhang/perl5/MSF-DTA/data/processed/models/model_' + model_st + '_' + dataset + '_' + cuda_name + '.pkl'
        # result_file_name = 'result_' + model_st + '_' + dataset +  '.csv'
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1)
            # print('predicting for valid data')
            # print('predicting for test data')
            G, P = predicting(model, device, valid_loader)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P), ci(G, P)]
            best_test_mse = ret[1]
            best_test_ci = ret[-1]
            print('Epoch: ', epoch + 1, 'best_valid_mse,best_valid_ci:', best_test_mse, best_test_ci, model_st, dataset)
            # val = mse(G,P)
            if best_test_mse < best_mse:
                best_mse = best_test_mse
                best_ci = best_test_ci
                best_epoch = epoch + 1

                print('rmse improved at epoch ', best_epoch, '; best_valid_mse,best_valid_ci:', best_mse, best_ci,
                      model_st, dataset)
                print()
            else:
                print('No improvement since epoch ', best_epoch, '; best_valid_mse,best_valid_ci:', best_mse, best_ci,
                      model_st, dataset)
                print()



