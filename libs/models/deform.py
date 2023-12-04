import torch
from torch import nn
import numpy as np

from libs.embeders.embedder_high import get_embedder
from libs.utils.network_utils import MotionBasisComputer
from libs.models.deconv_vol_decoder import MotionWeightVolumeDecoder
from libs.models.nonrigid import NonRigidMotionMLP
from libs.models.pose_refine import BodyPoseRefiner
from libs.models.offset import Offset

class Deformer(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=[4],
                 cond_in=[0],
                 cond_dim=96,
                 multires=0,
                 bias=1.0,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 **kwargs):
        super(Deformer, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.cond_in = cond_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in cond_in:
                lin = nn.Linear(dims[l] + cond_dim, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, p, c, **kwargs):
        batch_size, n_pts, _ = p.size()
        coords = p.reshape(-1, 3)

        if c.nelement() > 0:
            _, n_cond = c.size()
            input_cond = c.unsqueeze(1).expand(batch_size, n_pts, n_cond).view(batch_size*n_pts, n_cond)
        else:
            input_cond = torch.empty(batch_size * n_pts, 0, device=coords.device, dtype=coords.dtype)

        coords = coords
        if self.embed_fn_fine is not None:
            coords_embedded = self.embed_fn_fine(coords)
        else:
            coords_embedded = coords

        x = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.cond_in:
                x = torch.cat([x, input_cond], 1)

            if l in self.skip_in:
                x = torch.cat([x, coords_embedded], 1)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        out = x
        return out.reshape(batch_size, n_pts, -1)