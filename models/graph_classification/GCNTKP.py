from torch_geometric.nn import TopKPooling, GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool

import torch
from torch.nn import Linear, functional as F

from models.registry import Model


class GCNConv3TPK(torch.nn.Module):
    def __init__(self, num_features, output_channels, nb_neurons=128, **kwargs):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GCNConv3TPK, self).__init__()

        self.conv1 = GCNConv(num_features, nb_neurons)
        self.pool1 = TopKPooling(nb_neurons, ratio=0.8)
        self.conv2 = GCNConv(nb_neurons, nb_neurons)
        self.pool2 = TopKPooling(nb_neurons, ratio=0.8)
        self.conv3 = GCNConv(nb_neurons, nb_neurons)
        self.pool3 = TopKPooling(nb_neurons, ratio=0.8)

        self.lin1 = torch.nn.Linear(nb_neurons, 64)
        self.lin2 = torch.nn.Linear(64, output_channels)

    def forward(self, data, target_size=None, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)

        x = global_mean_pool(x, batch, size=target_size)

        x = F.relu(self.lin1(x))
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


class GCNConv1TPK(torch.nn.Module):
    def __init__(self, num_features, output_channels, nb_neurons=128, **kwargs):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GCNConv1TPK, self).__init__()

        self.conv1 = GCNConv(num_features, nb_neurons)
        self.conv2 = GCNConv(nb_neurons, nb_neurons)
        self.pool = TopKPooling(nb_neurons, ratio=0.8)
        self.conv3 = GCNConv(nb_neurons, nb_neurons)

        self.lin1 = torch.nn.Linear(nb_neurons, 64)
        self.lin2 = torch.nn.Linear(64, output_channels)

    def forward(self, data, target_size=None, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch, size=target_size)
        # x = [global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.lin1(x))
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


class GCNConv0TPK(torch.nn.Module):
    def __init__(self, num_features, output_channels, nb_neurons=128, **kwargs):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GCNConv0TPK, self).__init__()

        self.conv1 = GCNConv(num_features, nb_neurons)
        self.conv2 = GCNConv(nb_neurons, nb_neurons)
        self.conv3 = GCNConv(nb_neurons, nb_neurons)

        self.lin1 = torch.nn.Linear(nb_neurons, 64)
        self.lin2 = torch.nn.Linear(64, output_channels)

    def forward(self, data, target_size=None, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch, size=target_size)

        x = F.relu(self.lin1(x))
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


@Model
def gcnconv0TPK(num_features, output_channels, **kwargs):
    """
    Simple Graph Convolution Neural Network
    """
    return GCNConv0TPK(num_features, output_channels, **kwargs)


@Model
def gcnconv1TPK(num_features, output_channels, **kwargs):
    """
    Simple Graph Convolution Neural Network
    """
    return GCNConv1TPK(num_features, output_channels, **kwargs)


@Model
def gcnconv3TPK(num_features, output_channels, **kwargs):
    """
    Simple Graph Convolution Neural Network
    """
    return GCNConv3TPK(num_features, output_channels, **kwargs)

