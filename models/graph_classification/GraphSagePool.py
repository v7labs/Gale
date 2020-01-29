from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool

import torch
from torch.nn import Linear, functional as F

from models.registry import Model


class GraphSage3TPK(torch.nn.Module):
    def __init__(self, num_features, output_channels):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GraphSage3TPK, self).__init__()

        self.conv1 = SAGEConv(num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, output_channels)

    def forward(self, data, target_size=None):
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


class GraphSage1TPK(torch.nn.Module):
    def __init__(self, num_features, output_channels):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GraphSage1TPK, self).__init__()

        self.conv1 = SAGEConv(num_features, 128)
        self.conv2 = SAGEConv(128, 128)
        self.pool = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)

        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, output_channels)

    def forward(self, data, target_size=None):
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


class GraphSage0TPK(torch.nn.Module):
    def __init__(self, num_features, output_channels):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GraphSage0TPK, self).__init__()

        self.conv1 = SAGEConv(num_features, 128)
        self.conv2 = SAGEConv(128, 128)
        self.conv3 = SAGEConv(128, 128)

        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, output_channels)

    def forward(self, data, target_size=None):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch, size=target_size)

        x = F.relu(self.lin1(x))
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


@Model
def graphsage0TPK(num_features, output_channels, **kwargs):
    """
    Simple Graph Convolution Neural Network
    """
    return GraphSage0TPK(num_features, output_channels)


@Model
def graphsage1TPK(num_features, output_channels, **kwargs):
    """
    Simple Graph Convolution Neural Network
    """
    return GraphSage1TPK(num_features, output_channels)


@Model
def graphsage3TPK(num_features, output_channels, **kwargs):
    """
    Simple Graph Convolution Neural Network
    """
    return GraphSage3TPK(num_features, output_channels)

