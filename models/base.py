import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList, BatchNorm1d
from torch_sparse import SparseTensor
from pathlib import Path
import pdb

from tqdm import tqdm


class BaseModel(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False, use_linear=False):

        super(BaseModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.dropout = torch.nn.Dropout(p=dropout)
        self.activation = torch.nn.ReLU()
        self.batch_norm = batch_norm
        self.residual = residual
        self.num_layers = num_layers
        self.use_linear = use_linear
        if self.batch_norm:
            self.bns = ModuleList()
            for _ in range(num_layers - 1):
                bn = BatchNorm1d(hidden_channels)
                self.bns.append(bn)


    @classmethod
    def from_pretrained(cls, in_channels: int, out_channels: int, saved_ckpt_path: str, **kwargs):
        model = cls(in_channels=in_channels, out_channels=out_channels, **kwargs)
        #pdb.set_trace()
        if not saved_ckpt_path.endswith('.pt'):
            checkpoints = [str(x) for x in Path(saved_ckpt_path).glob(f"{cls.__name__}_*.pt")]
            if '_Lora' in cls.__name__:
                checkpoints = [str(x) for x in Path(saved_ckpt_path).glob(f"{cls.__name__.replace('_Lora', '')}_*.pt")]
            if '_MLP' not in cls.__name__:
                glob_checkpoints = [x for x in checkpoints if '_MLP' not in x]
            else:
                glob_checkpoints = checkpoints
            # print(checkpoints)

            # checkpoints = [str(x) for x in Path(saved_ckpt_path).glob(f"{cls.__name__}_run*.pt")]
            # glob_checkpoints = checkpoints
            assert len(glob_checkpoints) == 1
            saved_ckpt_path = glob_checkpoints[0]
        print(f'load model weights from {saved_ckpt_path}')
        state_dict = torch.load(saved_ckpt_path, map_location='cpu')
        final_state_dict = {}
        ignore_keys = ['edit_lrs']
        for k, v in state_dict.items():
            if k in ignore_keys:
                continue
            if k.startswith('model'):
                new_k = k.split('model.')[1]
                final_state_dict[new_k] = v
            else:
                final_state_dict[k] = v
        model.load_state_dict(final_state_dict, strict=False)
        return model


    def reset_parameters(self):
        raise NotImplementedError


class BaseGNNModel(BaseModel):
    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0,
                 batch_norm: bool = False, residual: bool = False, use_linear=False):
        super(BaseGNNModel, self).__init__(in_channels, hidden_channels, out_channels, num_layers,
                                           dropout, batch_norm, residual, use_linear)
        if self.use_linear:
            self.lins = torch.nn.ModuleList()
        self.convs = ModuleList()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bns:
                bn.reset_parameters()


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
