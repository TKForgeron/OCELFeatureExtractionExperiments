import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

torch.manual_seed(42)

import torch
from torch.nn import Sequential, MSELoss
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_hidden_features):
        super().__init__()

        self.gconv1 = GCNConv(num_node_features, num_hidden_features)
        self.gconv2 = GCNConv(num_hidden_features, num_hidden_features)
        self.out = Linear(num_hidden_features, 1)

    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index

        # First Message Passing Layer (Transformation)
        x = self.gconv1(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.gconv2(x, edge_index)
        x = x.relu()
        # x = F.dropout(x, p=0.5, training=self.training)

        # Output layer
        # x = F.mse_loss()
        # x = tf.reshape(h, (int(h.shape[0] / 4), (4 * 24))) MAKE THIS IN PYTORCH?
        out = self.out(x)
        return out
