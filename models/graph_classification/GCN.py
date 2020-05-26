import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, global_mean_pool, JumpingKnowledge

from models.registry import Model


class GCN(torch.nn.Module):
    def __init__(self, num_features, output_channels, num_layers=3, nb_neurons=128, **kwargs):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, nb_neurons)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(nb_neurons, nb_neurons))
        self.lin1 = Linear(nb_neurons, nb_neurons)
        self.lin2 = Linear(nb_neurons, output_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, target_size, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch, size=target_size)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


class GCNWithJK(torch.nn.Module):
    def __init__(self, num_features, output_channels, num_layers=3, nb_neurons=128, mode='cat', **kwargs):
        super(GCNWithJK, self).__init__()
        self.conv1 = GCNConv(num_features, nb_neurons)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(nb_neurons, nb_neurons))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * nb_neurons, nb_neurons)
        else:
            self.lin1 = Linear(nb_neurons, nb_neurons)
        self.lin2 = Linear(nb_neurons, output_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data, target_size, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        xs = [x]
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs += [x]
        x = self.jump(xs)
        x = global_mean_pool(x, batch, size=target_size)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


@Model
def gcn(num_features, output_channels, **kwargs):
    """
    GCN.py implementation from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
    """
    return GCN(num_features, output_channels, **kwargs)


@Model
def gcnJK(num_features, output_channels, **kwargs):
    """
    GCN.py implementation with JK from https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gcn.py
    """
    return GCNWithJK(num_features, output_channels, **kwargs)