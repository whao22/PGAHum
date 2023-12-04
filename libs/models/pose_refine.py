import torch.nn as nn
import torch

from libs.utils.network_utils import initseq, RodriguesModule


class BodyPoseRefiner(nn.Module):
    def __init__(self,
                 embedding_size=69,
                 mlp_width=256,
                 mlp_depth=4,
                 total_bones=24,
                 **_):
        super(BodyPoseRefiner, self).__init__()
        self.progress = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)  # use Parameter so it could be checkpointed
        
        block_mlps = [nn.Linear(embedding_size, mlp_width), nn.ReLU()]
        
        for _ in range(0, mlp_depth-1):
            block_mlps += [nn.Linear(mlp_width, mlp_width), nn.ReLU()]

        self.total_bones = total_bones - 1
        block_mlps += [nn.Linear(mlp_width, 3 * self.total_bones)]

        self.block_mlps = nn.Sequential(*block_mlps)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope the rotation matrix can be identity 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

        self.rodriguez = RodriguesModule()

    def forward(self, pose_input):
        """_summary_

        Args:
            pose_input (_type_): _description_

        Returns:
            _type_: _description_
        """
        rvec = self.block_mlps(pose_input).view(-1, 3) # (B*23, 3)
        Rs = self.rodriguez(rvec).view(-1, self.total_bones, 3, 3) # (B, 23, 3, 3)
        
        return {
            "Rs": Rs
        }
