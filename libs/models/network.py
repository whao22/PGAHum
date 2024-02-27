import torch
import numpy as np
from libs.embeders.embedder_nerf import get_embedder

""" MLP for neural implicit shapes. The code is based on https://github.com/lioryariv/idr with adaption. """
class ImplicitNetwork(torch.nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=[],
        cond_in=[],
        cond_dim=69,
        multires=0,
        bias=1.0,
        geometric_init=True,
        weight_norm=True,
        dim_cond_embed=-1,
        **kwargs,
    ):
        super().__init__()

        dims = [d_in] + [d_hidden] * n_layers + [d_out]
        self.num_layers = len(dims)

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.cond_layer = cond_in
        self.cond_dim = cond_dim

        self.dim_cond_embed = dim_cond_embed
        if dim_cond_embed > 0:
            self.lin_p0 = torch.nn.Linear(self.cond_dim, dim_cond_embed)
            self.cond_dim = dim_cond_embed

        self.skip_layer = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_layer:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in self.cond_layer:
                lin = torch.nn.Linear(dims[l] + self.cond_dim, out_dim)
            else:
                lin = torch.nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(
                        lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001
                    )
                    torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_layer:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = torch.nn.Softplus(beta=100)

    def forward(self, input, cond, mask=None):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            N: number of points
            D: input dimension
            
        Args:
            input (tensor): network input. shape: [B, N, D]
            cond (dict): conditional input.
            mask (tensor, optional): only masked inputs are fed into the network. shape: [B, N]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [N, D, ?]
        """


        n_batch, n_point, n_dim = input.shape

        if n_batch * n_point == 0:
            return input

        # reshape to [N,?]
        input = input.reshape(n_batch * n_point, n_dim)
        if mask is not None:
            input = input[mask]

        input_embed = input if self.embed_fn is None else self.embed_fn(input)

        if len(self.cond_layer):
            cond = cond["smpl"]
            n_batch, n_cond = cond.shape
            input_cond = cond.unsqueeze(1).expand(n_batch, n_point, n_cond)
            input_cond = input_cond.reshape(n_batch * n_point, n_cond)

            if mask is not None:
                input_cond = input_cond[mask]

            if self.dim_cond_embed > 0:
                input_cond = self.lin_p0(input_cond)

        x = input_embed

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.cond_layer:
                x = torch.cat([x, input_cond], dim=-1)

            if l in self.skip_layer:
                x = torch.cat([x, input_embed], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        # add placeholder for masked prediction
        if mask is not None:
            x_full = torch.zeros(n_batch * n_point, x.shape[-1], device=x.device)
            x_full[mask] = x
        else:
            x_full = x

        return x_full.reshape(n_batch, n_point, -1)