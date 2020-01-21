from torch_geometric.nn import TopKPooling, GraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool

import torch
from torch.nn import Linear, functional as F

from models.registry import Model


class GraphConv3TKP(torch.nn.Module):
    def __init__(self, num_features, output_channels):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GraphConv3TKP, self).__init__()

        self.conv1 = GraphConv(num_features, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, output_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class GraphConv1TKP(torch.nn.Module):
    def __init__(self, num_features, output_channels):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GraphConv1TKP, self).__init__()

        self.conv1 = GraphConv(num_features, 128)
        self.conv2 = GraphConv(128, 128)
        self.pool = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, output_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


class GraphConv0TPK(torch.nn.Module):
    def __init__(self, num_features, output_channels):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GraphConv0TPK, self).__init__()

        self.conv1 = GraphConv(num_features, 128)
        self.conv2 = GraphConv(128, 128)
        self.conv3 = GraphConv(128, 128)

        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, output_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_max_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x

class GraphConv0TPK_(torch.nn.Module):
    def __init__(self, num_features, output_channels):
        """

        Parameters
        ----------
        num_features: int
            number of node features
        output_channels: int
            number of classes
        """
        super(GraphConv0TPK, self).__init__()

        self.conv1 = GraphConv(num_features, 128)
        self.conv2 = GraphConv(128, 128)
        self.conv3 = GraphConv(128, 128)

        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, output_channels)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x1 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x2 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x3 = torch.cat([global_max_pool(x, batch), global_mean_pool(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

#
# @Model
# def graphconv0TPK(num_features, output_channels, **kwargs):
#     """
#     Simple Graph Convolution Neural Network
#     """
#     return GraphConv0TPK(num_features, output_channels)
#
#
# @Model
# def graphconv1TKP(num_features, output_channels, **kwargs):
#     """
#     Simple Graph Convolution Neural Network
#     """
#     return GraphConv1TKP(num_features, output_channels)
#
#
# @Model
# def graphconv3TKP(num_features, output_channels, **kwargs):
#     """
#     Simple Graph Convolution Neural Network
#     """
#     return GraphConv3TKP(num_features, output_channels)

