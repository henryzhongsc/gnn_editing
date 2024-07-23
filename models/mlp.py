import torch
from torch.nn import BatchNorm1d
from torch import Tensor
from .base import BaseModel


class MLP(BaseModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False,
                 load_pretrained_backbone: bool = False,
                 saved_ckpt_path: str = ''):

        super(MLP, self).__init__(in_channels, hidden_channels, out_channels, 
                                  num_layers, dropout, batch_norm, residual)
        self.lins = torch.nn.ModuleList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0:
                in_dim = in_channels
            if i == num_layers - 1:
                out_dim = out_channels
            lin = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=True)
            self.lins.append(lin)


    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()


    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        for idx in range(self.num_layers - 1):
            lin = self.lins[idx]
            h = lin(x, *args, **kwargs)
            if self.batch_norm:
                # import pdb; pdb.set_trace()
                h = self.bns[idx](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            x = self.activation(h)
            x = self.dropout(x)
        x = self.lins[-1](x, *args, **kwargs)
        return x