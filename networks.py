import torch
from torch_geometric.nn import GCNConv, MessagePassing
from torch.nn import Sequential, Linear, ReLU
from torch import Tensor


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr="max")  # "Max" aggregation.
        self.mlp = Sequential(
            Linear(2 * in_channels, 64),
            ReLU(),
            Linear(64, out_channels),
        )

    def reset_parameters(self):
        for i, l in enumerate(self.mlp):
            if type(l) == Linear:
                torch.nn.init.xavier_normal_(l.weight)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        return self.propagate(edge_index, x=x)  # shape [num_nodes, out_channels]

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        # x_j: Source node features of shape [num_edges, in_channels]
        # x_i: Target node features of shape [num_edges, in_channels]
        # edge_features = torch.cat([x_i, x_j - x_i], dim=-1)     # bad
        edge_features = torch.cat([x_i, x_j], dim=-1)
        return self.mlp(edge_features)  # shape [num_edges, out_channels]


class HGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = EdgeConv(in_channels, hidden_channels)
        self.conv2 = EdgeConv(hidden_channels, out_channels)
        self.readout = Linear(2 * out_channels, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x = self.conv1(x, edge_index)  # .relu()
        x = self.conv2(x, edge_index)
        x = self.readout(x.T)
        return x
