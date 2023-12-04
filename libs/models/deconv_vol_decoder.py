import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.utils.network_utils import ConvDecoder3D


class MotionWeightVolumeDecoder(nn.Module):
    def __init__(self, embedding_size=256, volume_size=32, total_bones=24):
        super(MotionWeightVolumeDecoder, self).__init__()

        self.progress = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)  # use Parameter so it could be checkpointed
        self.total_bones = total_bones
        self.volume_size = volume_size
        
        self.const_embedding = nn.Parameter(
            torch.randn(embedding_size), requires_grad=True 
        )

        self.decoder = ConvDecoder3D(
            embedding_size=embedding_size,
            volume_size=volume_size, 
            voxel_channels=total_bones+1)


    def forward(self, motion_weights_priors): # (B, 24+1, 32, 32, 32)
        embedding = self.const_embedding[None, ...] # (B, 256)
        decoded_weights =  F.softmax(self.decoder(embedding) + \
                                        torch.log(motion_weights_priors), 
                                     dim=1)
        
        return decoded_weights
