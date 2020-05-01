from torch_geometric.nn import TopKPooling, GraphConv
from torch_geometric.nn import global_mean_pool

import torch
from torch.nn import Linear, functional as F

from models.registry import Model


class GraphConv3TPK(torch.nn.Module):
    def __init__(self, num_features, output_channels, nb_neurons=128, **kwargs):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GraphConv3TPK, self).__init__()

        self.conv1 = GraphConv(num_features, nb_neurons)
        self.pool1 = TopKPooling(nb_neurons, ratio=0.8)
        self.conv2 = GraphConv(nb_neurons, nb_neurons)
        self.pool2 = TopKPooling(nb_neurons, ratio=0.8)
        self.conv3 = GraphConv(nb_neurons, nb_neurons)
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


class GraphConv1TPK(torch.nn.Module):
    def __init__(self, num_features, output_channels, nb_neurons=128, **kwargs):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GraphConv1TPK, self).__init__()

        self.conv1 = GraphConv(num_features, nb_neurons)
        self.conv2 = GraphConv(nb_neurons, nb_neurons)
        self.pool = TopKPooling(nb_neurons, ratio=0.8)
        self.conv3 = GraphConv(nb_neurons, nb_neurons)

        self.lin1 = torch.nn.Linear(nb_neurons, 64)
        self.lin2 = torch.nn.Linear(64, output_channels)

    def forward(self, data, target_size=None, **kwargs):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch, size=target_size)

        x = F.relu(self.lin1(x))
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


class GraphConv0TPK(torch.nn.Module):
    def __init__(self, num_features, output_channels, nb_neurons=128, **kwargs):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GraphConv0TPK, self).__init__()

        self.conv1 = GraphConv(num_features, nb_neurons)
        self.conv2 = GraphConv(nb_neurons, nb_neurons)
        self.conv3 = GraphConv(nb_neurons, nb_neurons)

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
def graphconv0TPK(num_features, output_channels, **kwargs):
    """
    Simple Graph Convolution Neural Network
    """
    return GraphConv0TPK(num_features, output_channels, **kwargs)


@Model
def graphconv1TPK(num_features, output_channels, **kwargs):
    """
    Simple Graph Convolution Neural Network
    """
    return GraphConv1TPK(num_features, output_channels, **kwargs)


@Model
def graphconv3TPK(num_features, output_channels, **kwargs):
    """
    Simple Graph Convolution Neural Network
    """
    return GraphConv3TPK(num_features, output_channels, **kwargs)

