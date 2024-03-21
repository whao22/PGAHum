import torch
import pytorch3d.ops as ops

import os
import sys
sys.path.append('.')
import pickle
import numpy as np
from tqdm import tqdm
from pyhocon import ConfigFactory
from torch.utils.tensorboard import SummaryWriter

from libs.module_config import get_skinning_model
from libs.utils.general_utils import get_02v_bone_transforms, hierarchical_softmax

def backward_lbs(points_obs, dst_gtfms, pts_weights):
    ''' Backward skinning based on neural network predicted skinning weights 
    Args:
        points_obs (tensor): canonical points. shape: [B, N, D]
        pts_weights (tensor): conditional input. [B, N, J]
        dst_gtfms (tensor): bone transformation matrices. shape: [B, J, D+1, D+1]
    Returns:
        x (tensor): skinned points. shape: [B, N, D]
    '''
    batch_size, N_points, _ = points_obs.shape
    device = points_obs.device
    
    transforms_fwd = torch.matmul(pts_weights, dst_gtfms.reshape(batch_size, -1, 16)).reshape(batch_size, N_points, 4, 4)
    transforms_bwd = torch.inverse(transforms_fwd)
    # transforms_bwd = FastDiff4x4MinvFunction.apply(transforms_fwd.reshape(-1, 4, 4))[0].reshape_as(transforms_fwd)
    homogen_coord = torch.ones(batch_size, N_points, 1, dtype=torch.float32, device=device)
    points_homo = torch.cat([points_obs, homogen_coord], dim=-1).reshape(batch_size, N_points, 4, 1)
    points_cnl = torch.matmul(transforms_bwd, points_homo)[..., :3, 0]
    
    return points_cnl

def sample_points_weights(points_obs, skinning_weights, dst_vertices):
    ''' Sample conditional input for skinning model
    Args:
        points_obs (tensor): canonical points. shape: [B, N, D]
    Returns:
        pts_weights (tensor): conditional input. [B, N, J]
    '''
    batch_size, N_points, _ = points_obs.shape
    device = points_obs.device
    N_knn = 5
    
    # sample skinning weights from SMPL prior weights
    knn_ret = ops.knn_points(points_obs, dst_vertices, K=N_knn)
    p_idx, p_squared_dists = knn_ret.idx, knn_ret.dists
    p_dists = p_squared_dists**0.5
    
    # point-wise weights based on distance
    w = p_dists.sum(-1, True) / (p_dists + 1e-8)
    w = w / (w.sum(-1, True) + 1e-8)
    
    # # point-wise decay factor
    # decay_factor = self.decay_from_dists(p_dists)
    bv, _ = torch.meshgrid([torch.arange(batch_size).to(device), torch.arange(N_points).to(device)], indexing='ij')
    pts_W_sampled = 0.
    for i in range(N_knn):
        # pts_W_sampled += skinning_weights[bv, p_idx[..., i], :] * w[..., i:i+1] * decay_factor[..., i:i+1]
        pts_W_sampled += skinning_weights[bv, p_idx[..., i], :] * w[..., i:i+1]
    
    return pts_W_sampled

def loss_func(pts_w_pre, pts_w_gt):
    return torch.abs(pts_w_pre - pts_w_gt).sum(-1).mean()

def softmax_func(wi):
    wi = wi * 20
    if wi.size(-1) == 24:
        w_ret = torch.softmax(wi, dim=-1) # naive softmax
    elif wi.size(-1) == 25:
        w_ret = hierarchical_softmax(wi) # hierarchical softmax in SNARF
    else:
        raise ValueError('Wrong output size of skinning network. Expected 24 or 25, got {}.'.format(wi.size(-1)))
    return w_ret

def training_skinning_model(skinning_model, skinning_weights, vertices_obs, vertices_cnl, dst_gtfms):
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(logs_dir)
    
    optimizer = torch.optim.Adam(skinning_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)
    
    for step in tqdm(range(10000)):
        optimizer.zero_grad()
        
        points = (torch.rand(1, 500_000, 3).to(vertices_cnl)-0.5)*2
        pts_weights_gt = sample_points_weights(points, skinning_weights, vertices_cnl)
        pts_weights_pred_raw = skinning_model(points, torch.empty(points.size(0), 0, device=points.device, dtype=torch.float32))
        pts_weights_pred = softmax_func(pts_weights_pred_raw)
        loss = loss_func(pts_weights_pred, pts_weights_gt)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if step % 1000 == 0:
            torch.save(skinning_model.state_dict(), os.path.join(ckpt_dir, f"skinning_model_{step}.pth"))
        
        if step % 10 == 0:
            writer.add_scalar('loss', loss.item(), step)
            print(f"Epoch {step}: loss {loss.item()}")

def test_skinning_model(skinning_model, vertices_obs, vertices_cnl, dst_gtfms):
    skinning_model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'checkpoints/skinning_model_9000.pth')))
    import trimesh
    with torch.no_grad():
        pts_weights_pred_raw = skinning_model(vertices_cnl, torch.empty(vertices_cnl.size(0), 0, device=vertices_cnl.device, dtype=torch.float32))
        pts_weights_pred = softmax_func(pts_weights_pred_raw[..., :24])
        vertices_warp = backward_lbs(vertices_obs, dst_gtfms, pts_weights_pred)
        trimesh.Trimesh(vertices_warp.reshape(-1, 3).detach().cpu().numpy()).export("vertices_warp.obj")

if __name__ == '__main__':
    # load config
    conf = ConfigFactory.parse_file('confs/hfavatar-zjumocap/ZJUMOCAP-377-mono-4gpus.conf')
    device = torch.device('cuda:0')
    
    # load SMPL prior weights and canonical joints
    skinning_weights = dict(np.load('data/body_models/misc/skinning_weights_all.npz'))['neutral']
    skinning_weights = torch.from_numpy(skinning_weights).float()[None, ...].to(device)
    
    with open("data/data_prepared/CoreView_377/canonical_joints.pkl", 'rb') as f:
            cl_joint_data = pickle.load(f)
    vertices_obs = cl_joint_data['vertices']
    vertices_obs = torch.from_numpy(vertices_obs).float()[None, ...].to(device)
    
    joints_tpose = cl_joint_data['joints']
    dst_gtfms = get_02v_bone_transforms(joints_tpose)
    dst_gtfms = torch.from_numpy(dst_gtfms).float()[None, ...].to(device)
    dst_gtfms = torch.inverse(dst_gtfms)
    
    vertices_cnl = backward_lbs(vertices_obs, dst_gtfms, skinning_weights).to(device)
    
    # Load skinning model and train
    skinning_model = get_skinning_model(conf, False).to(device)
    # training_skinning_model(skinning_model, skinning_weights, vertices_obs, vertices_cnl, dst_gtfms)
    test_skinning_model(skinning_model, vertices_obs, vertices_cnl, dst_gtfms)