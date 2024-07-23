from tqdm import tqdm

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d
from torch_sparse import SparseTensor
from torch_geometric.nn import SAGEConv
from .base import BaseGNNModel


class SAGE(BaseGNNModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False, use_linear=False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):
        super(SAGE, self).__init__(in_channels, hidden_channels, out_channels, 
                                  num_layers, dropout, batch_norm, residual, use_linear)
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
            conv = SAGEConv(in_dim, out_dim)
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