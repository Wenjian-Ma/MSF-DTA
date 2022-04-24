import os
from math import sqrt
from scipy import stats
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
import scipy.sparse as sp
from sklearn.preprocessing import scale
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np



def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def load_ppi_network(filename, gene_num, thr):
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num, gene_num))
    for x in tqdm(data):
        temp = x.strip().split("\t")
        # check whether score larger than the threshold
        if float(temp[2]) >= thr:
            adj[int(temp[0]), int(temp[1])] = 1
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T

    return adj


def load_simi_network(filename, gene_num, thr):
    with open(filename) as f:
        data = f.readlines()
    adj = np.zeros((gene_num, gene_num))
    for x in tqdm(data):
        temp = x.strip().split("\t")
        # check whether evalue smaller than the threshold
        if float(temp[2]) <= thr:
            adj[int(temp[0]), int(temp[1])] = 1
    if (adj.T == adj).all():
        pass
    else:
        adj = adj + adj.T

    return adj


def load_data(graph_type, uniprot, args):
    print('loading data...')

    def reshape(features):
        return np.hstack(features).reshape((len(features), len(features[0])))

    # get feature representations
    features_seq = scale(reshape(uniprot['CT'].values))  # CT特征编码
    features_loc = reshape(uniprot['Sub_cell_loc_encoding'].values)  # 亚细胞位置编码
    features_domain = reshape(uniprot['Pro_domain_encoding'].values)  # 蛋白结构域编码
    #features_domain = np.load('./data/human/domain_feature.npy')


    print('generating features...')
    # attribute=6
    if graph_type == "ppi":
        attribute = args.ppi_attributes
    elif graph_type == "sequence_similarity":
        attribute = args.simi_attributes

    if attribute == 0:
        features = np.identity(uniprot.shape[0])
        print("Without features")
    elif attribute == 1:
        features = features_seq
        print("Only use sequence feature")
    elif attribute == 2:
        features = features_loc
        print("Only use location feature")
    elif attribute == 3:
        features = features_domain
        print("Only use domain feature")
    elif attribute == 5:
        features = np.concatenate((features_loc, features_domain), axis=1)
        print("use location and domain features")
    elif attribute == 6:
        features = np.concatenate((features_seq, features_loc, features_domain), axis=1)  # 三种特征进行拼接
        print("Use all the features")
    elif attribute == 7:
        features = np.concatenate((features_seq, features_loc, features_domain, np.identity(uniprot.shape[0])), axis=1)
        print("Use all features plus identity")

    features = sp.csr_matrix(features)

    print('loading graph...')
    if graph_type == "sequence_similarity":
        filename =  "./data/networks/sequence_similarity.txt"
        adj = load_simi_network(filename, uniprot.shape[0], args.thr_evalue)
    elif graph_type == "ppi":
        filename = "./data/networks/ppi.txt"
        adj = load_ppi_network(filename, uniprot.shape[0], args.thr_combined)

    adj = sp.csr_matrix(adj)

    return adj, features


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None,seq = None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)

        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph,seq)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y,smile_graph,seq):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            seq_prot = seq[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]

            molecule = Chem.MolFromSmiles(smiles)
            fingerprints = MACCSkeys.GenMACCSKeys(molecule)
            fingerprint = np.array([int(i) for i in list(fingerprints.ToBitString())]).reshape((1, 167))

            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]),
                                fingerprint=torch.FloatTensor(fingerprint))
            GCNData.target = torch.FloatTensor([target])
            GCNData.prot_seq = torch.LongTensor([seq_prot])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci