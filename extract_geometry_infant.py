import os
import argparse
from pyhocon import ConfigFactory
import torch
import numpy as np
import time
import trimesh
import mcubes

from libs import module_config
from libs.utils.general_utils import sample_sdf_from_grid

# Arguments
parser = argparse.ArgumentParser(description='Extract Geometry from SDF Network.')
parser.add_argument('--conf', type=str, help='Path to config file.', default="confs/hfavatar-zju/ZJUMOCAP-377-mono-4gpus.conf")
parser.add_argument('--base_exp_dir', type=str, default="exp/CoreView_377_1702989327_slurm_w/_inner-w/_init_sdf-w/_tri")
parser.add_argument('--resolution', type=int, default=1024)
parser.add_argument('--mcthreshold', type=float, default=0.0)
parser.add_argument('--device', type=str, default="cuda:2", help="cuda / cpu")

def lambda_sdf(points, sdf_network, sdf_kwargs, use_init_sdf=True, scale=1):
    """Query the sdf of points. The term 'scale' means scale of scaling of delta sdf.

    Args:
        points (Tensor): points to be queried.
        sdf_network (_type_): _description_
        sdf_kwargs (_type_): _description_
        use_init_sdf (bool, optional): _description_. Defaults to True.
        scale (int or ndarray, optional): Defauts to 1 in training time, in test it 
                                          may be set to ndarray ofor each points.

    Returns:
        sdf: sdf of points
    """
    sdf_nn_output = sdf_network(points)
    sdf = sdf_nn_output[:, :1]
    
    if use_init_sdf:
        init_sdf = sample_sdf_from_grid(points, **sdf_kwargs)
        sdf = init_sdf + sdf * scale
        
    return sdf

def extract_fields(bound_min, bound_max, resolution, sdf_network, sdf_kwargs, scale, device):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).to(device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).to(device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).to(device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    points = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = lambda_sdf(points, sdf_network, sdf_kwargs, scale).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, sdf_network, sdf_kwargs, scale=1, device="cpu"):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, sdf_network, sdf_kwargs, scale, device)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


if  __name__ == '__main__':
    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.conf)
    out_dir = args.base_exp_dir
    resolution = args.resolution
    threshold = args.mcthreshold
    device = torch.device(args.device)
    
    # Model
    print("Load model ...")
    model = module_config.get_model(conf, args.base_exp_dir)

    # Load State Dict
    checkpoint_path = os.path.join(out_dir, 'checkpoints/last.ckpt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('No checkpoint is found!')
    print("Load state dict ...")
    model.load_state_dict(torch.load(checkpoint_path)['state_dict']).to(device)
    sdf_network = model.sdf_network

    # extract geometry
    smpl_sdf = np.load("data/body_models/smpl/infant/infant_sdf.npy", allow_pickle=True).item()
    bound_min = torch.tensor([-1.01, -1.01, -1.01]).to(device)
    bound_max = torch.tensor([1.01, 1.01, 1.01]).to(device)
    
    sdf_kwargs={
        "cnl_bbmin": smpl_sdf['bbmin'],
        "cnl_bbmax": smpl_sdf['bbmax'],
        "cnl_grid_sdf": torch.from_numpy(smpl_sdf['sdf_grid']).float().to(device),
    }
    print("Start extract geometry ...")
    vertices, triangles = extract_geometry(bound_min, bound_max, resolution, threshold, sdf_network, sdf_kwargs, scale=0.3, device=device)
    
    mesh_dir = os.path.join(out_dir, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(os.path.join(mesh_dir, f"{int(time.time())}.obj"))