import numpy as np
import trimesh
import mesh2sdf
import sys
sys.path.append(".")
import time
import pickle
import torch
import torch.nn.functional as F
from libs.utils.MCAcc.grid_sampler_mine import GridSamplerMine3dFunction
from libs.utils.grid_sample_3d import grid_sample_3d
from libs.utils.ops.grid_sample import grid_sample_3d as grid_sample_3d_cuda

def prepare_smpl_sdf():
    # load data
    with open("data/data_prepared/CoreView_377/0/canonical_joints.pkl", "rb") as f:
        data = pickle.load(f)
    vertices = data['vertices']
    faces = np.load("data/body_models/misc/faces.npz")['faces']

    # norm
    PADDING = 0.05
    maxv = abs(vertices).max()
    bbmax = maxv + PADDING
    bbmin = -maxv - PADDING
    # normlize
    vertices_norm = (vertices - bbmin) / (bbmax - bbmin + 1e-10)
    vertices_norm = (vertices_norm - 0.5 ) * 2 # [-1, 1]
    
    '''
    # unnormlize
    verties_ori = vertices_norm * 0.5 + 0.5
    verties_ori = verties_ori * (bbmax - bbmin + 1e-10) + bbmin
    '''
    
    s = time.time()
    sdf_grid, mesh = mesh2sdf.compute(vertices_norm, faces, size=128, return_mesh=True)
    e = time.time()
    
    # uniformly generate sampling points
    N_samples_per_dim = 128
    x = (np.linspace(0, 1, N_samples_per_dim) - 0.5 ) *2 # [-1, 1]
    y = (np.linspace(0, 1, N_samples_per_dim) - 0.5 ) *2
    z = (np.linspace(0, 1, N_samples_per_dim) - 0.5 ) *2
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    xyz = np.stack([xv, yv, zv], axis=-1)
    
    sdf_grid = sdf_grid.astype(np.float32)
    xyz = xyz.astype(np.float32)
    xyz_grid = xyz.copy()
    xyz_grid[..., 0] = xyz[..., 2]
    xyz_grid[..., 1] = xyz[..., 1]
    xyz_grid[..., 2] = xyz[..., 0]
    
    # sample sdf
    # sdf_grid_torch = torch.from_numpy(sdf_grid).permute([2, 0, 1])[None, None, ...] # (N, C, D, W, H)
    # xyz_torch = torch.from_numpy(xyz).permute([2, 0, 1, 3])[None, ...] # (N, D, H, W, 3)
    # sdf_sampled = F.grid_sample(sdf_grid_torch, xyz_torch)
    # sdf_sampled = sdf_sampled.permute([0, 1, 3, 4, 2]).squeeze().numpy()
    sdf_grid_torch = torch.from_numpy(sdf_grid)[None, None, ...] # (N, C, D, W, H)
    xyz_torch = torch.from_numpy(xyz_grid)[None, ...] # (N, D, H, W, 3)
    sdf_sampled = F.grid_sample(sdf_grid_torch, xyz_torch, mode='bilinear', padding_mode='border', align_corners=False)
    sdf_sampled2 = grid_sample_3d(sdf_grid_torch, xyz_torch)
    sdf_sampled3 = GridSamplerMine3dFunction.apply(sdf_grid_torch.cuda(), xyz_torch.cuda())
    sdf_sampled4 = grid_sample_3d_cuda(sdf_grid_torch, xyz_torch, padding_mode='border', align_corners=False)
    sdf_sampled = sdf_sampled.squeeze().numpy()
    sdf_sampled = sdf_sampled / 2 * (bbmax - bbmin)
    
    mask = sdf_sampled < 0
    x = xyz[mask]
    print(e - s)

if __name__=="__main__":
    prepare_smpl_sdf()