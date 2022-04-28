import torch
# from torch_geometric.nn import GCNConv
import torch.nn as nn
#from torch.nn.functional import adaptive


class InnerProductDecoder(nn.Module):
    def __init__(self, input_dim,act, dropout=0, **kwargs):
        super(InnerProductDecoder,self).__init__()
        # self.dropout = dropout
        self.act = act
        self.dropout = nn.Dropout(dropout)
    def forward(self, inputs):
        inputs = self.dropout(inputs)
        x = torch.transpose(inputs,0,1)
        x = torch.matmul(inputs, x)
        x = torch.reshape(x, [1,-1])
        outputs = self.act(x)
        return outputs


def dropout_sparse(x, keep_prob, num_nonzero_elems):#keep_prob=1
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += torch.rand(noise_shape)
    #dropout_mask = tf.cast(torch.floor(random_tensor), dtype=tf.bool)
    #pre_out = tf.sparse_retain(x, dropout_mask)
    return x


class GraphConvolutionSparse(nn.Module):
    def __init__(self,input_dim, output_dim,features_nonzero, dropout=0., act=nn.ReLU(), **kwargs):
        super(GraphConvolutionSparse,self).__init__()
        w = nn.Parameter(torch.Tensor(input_dim,output_dim))
        #w = torch.tensor(w,dtype=torch.long)
        self.weights = nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
        self.dropout = dropout
        #self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
    def forward(self,adj,inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = torch.matmul(x,self.weights)
        #print(torch.max(x))
        x = torch.matmul(adj,x)
        #print(torch.max(x))
        outputs = self.act(x)
        return outputs



class GraphConvolution(nn.Module):
    def __init__(self,input_dim, output_dim,dropout=0., act=nn.ReLU(), **kwargs):
        super(GraphConvolution, self).__init__()
        w = nn.Parameter(torch.Tensor(input_dim,output_dim))
        self.weights = nn.init.xavier_uniform_(w,gain=nn.init.calculate_gain('relu'))
        self.dropout = dropout
        #self.adj = adj
        self.act = act
    def forward(self,adj,inputs):
        x = inputs
        x = torch.matmul(x, self.weights)
        #x = tf.sparse_tensor_dense_matmul(self.adj, x)
        x = torch.matmul(adj,x)
        outputs = self.act(x)
        return outputs



class GCNModelVAE(torch.nn.Module):
    def __init__(self,num_features, num_nodes, features_nonzero, hidden1, hidden2,device, dropout=0, **kwargs):
        super(GCNModelVAE, self).__init__()
        self.n_samples = num_nodes
        self.hidden1_dim = hidden1
        self.hidden2_dim = hidden2
        self.input_dim = num_features
        # self.adj = adj
        # self.inputs = features
        self.device = device
        self.dropout = dropout
        self.features_nonzero = features_nonzero
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=self.hidden1_dim,
                                              features_nonzero=self.features_nonzero,
                                              act=nn.ReLU(),
                                              dropout=self.dropout)
        self.z_mean = GraphConvolution(input_dim=self.hidden1_dim,
                                       output_dim=self.hidden2_dim,
                                       act=lambda x: x,
                                       dropout=self.dropout)
        self.z_log_std = GraphConvolution(input_dim=self.hidden1_dim,
                                          output_dim=self.hidden2_dim,
                                          act=lambda x: x,
                                          dropout=self.dropout)
        #self.relu = nn.ReLU()
        #self.dropout = nn.Dropout(dropout)
        self.InnerProductDecoder = InnerProductDecoder(input_dim=self.hidden2_dim,act = lambda x: x)
    def forward(self, adj,x):
        x = self.hidden1(adj,x)

        self.z_mean_value = self.z_mean(adj,x)

        self.z_log_std_value = self.z_log_std(adj,x)

        z = self.z_mean_value + torch.randn(self.n_samples, self.hidden2_dim,device=self.device) * torch.exp(self.z_log_std_value)

        reconstructions = self.InnerProductDecoder(z)
        return self.z_mean_value,reconstructions
