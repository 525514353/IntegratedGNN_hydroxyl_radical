import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter,LSTMCell
from torch import nn
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool, GINConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor

class nd_conv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        return self.lin2(x_j)

class molecular_conv(torch.nn.Module):
    r"""
    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input = None
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.nd_conv = nd_conv(hidden_channels, hidden_channels, edge_dim,
                               dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GINConv(nn.Linear(hidden_channels,hidden_channels))
        self.mol_lstm = LSTMCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.nd_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_lstm.reset_parameters()
        self.lin2.reset_parameters()

    def activations_hook(self, grad):
        self.final_conv_grads = grad

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Tensor) -> Tensor:
        """"""
        '''for explainablity'''
        h0, edge_index, edge_weight = x, edge_index, sum(edge_attr)
        h0.requires_grad = True
        self.input = h0

        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.nd_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            with torch.enable_grad():
                self.final_conv_acts = gru(h, x).relu_()
            self.final_conv_acts.register_hook(self.activations_hook)


        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(self.final_conv_acts, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((self.final_conv_acts, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out,h = self.mol_lstm(out,(h,h))

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')


if __name__=='__main__':
    model = molecular_conv(out_channels=1,  # active or inactive
                    in_channels=3, edge_dim=2,
                    hidden_channels=512, num_layers=3, num_timesteps=5,
                    dropout=0.05)
    x=torch.randn(size=(6,3))
    edge_index=torch.randint(low=0,high=3,size=(2,4))
    edge_attr=torch.randn(size=(4,2))
    batch=torch.tensor([0,0,0,0,1,1])
    print(model(x, edge_index, edge_attr,batch))
