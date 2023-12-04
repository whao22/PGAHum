from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from libs.embeders.embedder_high import get_embedder

class Offset(nn.Module):
    def __init__(self,
                 D=8,
                 W=256,
                 d_in=3,
                 multires=0,
                 output_ch=4,
                 skips=[4]):
        super(Offset, self).__init__()
        self.D = D
        self.W = W
        self.d_in = d_in
        self.input_ch = 3
        self.embed_fn = None
        self.embed_fn_view = None
        
        self.multires = multires
        self.progress = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)  # use Parameter so it could be checkpointed

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn = embed_fn
            self.input_ch = input_ch

        self.skips = skips

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D - 1)])

        self.output_linear = nn.Linear(W, output_ch)

    def forward(self, input_pts):
        if self.embed_fn is not None:
            input_pts_enc = self.embed_fn(input_pts)
            input_pts_enc, weight = coarse2fine(self.progress.data, input_pts_enc, self.multires)
            input_pts = torch.cat([input_pts, input_pts_enc], dim=-1)  # [B,...,6L+3]

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
        
        out = self.output_linear(h)
        return out
    
def coarse2fine(progress_data, inputs, L):  # [B,...,N]
    barf_c2f = [0.1, 0.5]
    if barf_c2f is not None:
        # set weights for different frequency bands
        start, end = barf_c2f
        alpha = (progress_data - start) / (end - start) * L
        k = torch.arange(L, dtype=torch.float32, device=inputs.device)
        weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(np.pi).cos_()) / 2
        # apply weights
        shape = inputs.shape
        input_enc = (inputs.view(-1, L, int(shape[1]/L)) * weight.tile(int(shape[1]/L),1).T).view(*shape)
    return input_enc, weight