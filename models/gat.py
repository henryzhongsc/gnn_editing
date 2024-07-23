from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
from .base import BaseGNNModel

from .mlp import MLP


class GAT(BaseGNNModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, heads:int = 8, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False, use_linear=False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        super(GAT, self).__init__(in_channels, hidden_channels, out_channels,
                                  num_layers, dropout, batch_norm, residual, use_linear)
        num_heads = heads
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels * (heads ** i)
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
                num_heads = 1
            conv = GATConv(in_dim, out_dim, heads = num_heads, dropout = dropout)
            self.convs.append(conv)
            if self.use_linear:
                self.lins.append(torch.nn.Linear(in_dim, out_dim, bias=False))


    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        for idx in range(self.num_layers - 1):
            conv = self.convs[idx]
            h = conv(x, adj_t)
            if self.use_linear:
                linear = self.lins[idx](x)
                h = h + linear
            if self.batch_norm:
                h = self.bns[idx](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = self.activation(h)
            x = self.dropout(x)
        h = self.convs[-1](x, adj_t, *args, **kwargs)
        if self.use_linear:
            linear = self.lins[-1](x)
            x = h + linear
        else:
            x = h
        return x


    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, size):
        if self.use_linear:
            raise NotImplementedError
        if layer != 0:
            x = self.dropout(x)
        x_target = x[:size[1]]
        h = self.convs[layer]((x, x_target), adj_t)
        if layer < self.num_layers - 1:
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = F.relu(h)
        return h


    @torch.no_grad()
    def mini_inference(self, x_all, loader):
        pbar = tqdm(total=x_all.size(0) * len(self.convs))
        pbar.set_description('Evaluating')
        for i in range(len(self.convs)):
            xs = []
            for batch_size, n_id, adj in loader:
                edge_index, _, size = adj.to('cuda')
                x = x_all[n_id].to('cuda')
                xs.append(self.forward_layer(i, x, edge_index, size).cpu())
                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all

class GAT_MLP(BaseGNNModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int,  heads:int = 8,
                 shared_weights: bool = True, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        super(GAT_MLP, self).__init__(in_channels, hidden_channels, out_channels,
                                  num_layers, dropout, batch_norm, residual)
        # self.alpha, self.theta = alpha, theta

        if load_pretrained_backbone:
            self.GAT = GAT.from_pretrained(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                saved_ckpt_path=saved_ckpt_path,
                num_layers=num_layers,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual)
        else:
            self.GAT = GAT(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, \
                            num_layers=num_layers, heads = heads, dropout=dropout, batch_norm=batch_norm, residual=residual)
        self.MLP = MLP(in_channels=in_channels, hidden_channels=hidden_channels,
                        out_channels=out_channels, num_layers=num_layers, dropout=dropout,
                        batch_norm=batch_norm, residual=residual)

        self.mlp_freezed = True
        if load_pretrained_backbone:
            self.freeze_layer(self.GAT, freeze=True)
            self.freeze_layer(self.GAT, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_module(train=True)
        self.gnn_output = None


    def reset_parameters(self):
        ### reset GAT parameters
        for conv in self.GAT.convs:
            conv.reset_parameters()
        if self.GAT.batch_norm:
            for bn in self.GAT.bns:
                bn.reset_parameters()

        ### reset MLP parameters
        for lin in self.MLP.lins:
            lin.reset_parameters()
        if self.MLP.batch_norm:
            for bn in self.MLP.bns:
                bn.reset_parameters()

    def freeze_layer(self, model, freeze=True):
        for name, p in model.named_parameters():
            p.requires_grad = not freeze

    def freeze_module(self, train=True):
        ### train indicates whether train/eval editable ability
        if train:
            self.freeze_layer(self.GAT, freeze=False)
            self.freeze_layer(self.MLP, freeze=True)
            self.mlp_freezed = True
        else:
            self.freeze_layer(self.GAT, freeze=True)
            self.freeze_layer(self.MLP, freeze=False)
            self.mlp_freezed = False

    def forward(self, x: Tensor, adj_t: SparseTensor, *args, **kwargs) -> Tensor:
        GAT_out = self.GAT(x, adj_t, *args)
        if self.mlp_freezed:
            x = GAT_out
        else:
            MLP_out = self.MLP(x, *args)
            x = GAT_out + MLP_out
        return x

    def fast_forward(self, x: Tensor, idx) -> Tensor:
        assert self.gnn_output is not None
        assert not self.mlp_freezed
        return self.gnn_output[idx.to(self.gnn_output.device)].to(x.device) + self.MLP(x)