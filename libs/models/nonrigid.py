import torch
import torch.nn as nn
from libs.utils.network_utils import initseq


class NonRigidMotionMLP(nn.Module):
    def __init__(self,
                 pos_embed_size=3,
                 condition_code_size=69,
                 mlp_width=128,
                 mlp_depth=6,
                 skips=None,
                 **kwargs):
        super(NonRigidMotionMLP, self).__init__()
        self.progress = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)  # use Parameter so it could be checkpointed
        self.skips = [4] if skips is None else skips
        
        block_mlps = [nn.Linear(pos_embed_size+condition_code_size, 
                                mlp_width), nn.ReLU()]
        
        layers_to_cat_inputs = []
        for i in range(1, mlp_depth):
            if i in self.skips:
                layers_to_cat_inputs.append(len(block_mlps))
                block_mlps += [nn.Linear(mlp_width+pos_embed_size, mlp_width), 
                               nn.ReLU()]
            else:
                block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        block_mlps += [nn.Linear(mlp_width, 3)]

        self.block_mlps = nn.ModuleList(block_mlps)
        initseq(self.block_mlps)

        self.layers_to_cat_inputs = layers_to_cat_inputs

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope non-rigid offsets are zeros 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

    # calculate points_cnl which transformed from points_skeleton by non-rigid transformation
    def expand_input(self, input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))
        
    def forward(self, pos_embed, pos_xyz, condition_code, viewdirs=None):
        condition_code = self.expand_input(condition_code, pos_embed.size(0))
        h = torch.cat([condition_code, pos_embed], dim=-1)
        if viewdirs is not None:
            h = torch.cat([h, viewdirs], dim=-1)

        for i in range(len(self.block_mlps)):
            if i in self.layers_to_cat_inputs:
                h = torch.cat([h, pos_embed], dim=-1)
            h = self.block_mlps[i](h)
        trans = h

        result = {
            'xyz': pos_xyz + trans,
            'offsets': trans
        }
        
        return result
