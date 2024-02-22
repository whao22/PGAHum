import torch

net_path = 'data/pretrained/humannerf/zju_mocap/377/latest.tar' 
aaa = torch.load(net_path)
print()