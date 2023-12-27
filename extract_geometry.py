import os
import argparse
from pyhocon import ConfigFactory
import torch
import numpy as np
import time
import trimesh

from libs import module_config
from libs.utils.geometry_utils import extract_geometry

# Arguments
parser = argparse.ArgumentParser(description='Extract Geometry from SDF Network.')
parser.add_argument('--conf', type=str, help='Path to config file.', default="confs/hfavatar-zju/ZJUMOCAP-377-mono-4gpus.conf")
parser.add_argument('--base_exp_dir', type=str, default="exp/CoreView_377_1702989327_slurm_w/_inner-w/_init_sdf-w/_tri")
parser.add_argument('--resolution', type=int, default=1024)
parser.add_argument('--mcthreshold', type=float, default=0.0)
parser.add_argument('--device', type=str, default="cuda:2", help="cuda / cpu")

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
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    sdf_network = model.sdf_network.to(device)

    # extract geometry
    smpl_sdf = np.load(os.path.join(conf.dataset.data_dir, conf.dataset.train_split[0], "smpl_sdf.npy"), allow_pickle=True).item()
    bound_min = torch.tensor([-1.01, -1.01, -1.01]).to(device)
    bound_max = torch.tensor([1.01, 1.01, 1.01]).to(device)
    
    sdf_kwargs={
        "cnl_bbmin": smpl_sdf['bbmin'],
        "cnl_bbmax": smpl_sdf['bbmax'],
        "cnl_grid_sdf": torch.from_numpy(smpl_sdf['sdf_grid']).float().to(device),
    }
    print("Start extract geometry ...")
    vertices, triangles = extract_geometry(bound_min, bound_max, resolution, threshold, sdf_network, sdf_kwargs, device=device)
    
    mesh_dir = os.path.join(out_dir, 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    mesh = trimesh.Trimesh(vertices, triangles)
    mesh.export(os.path.join(mesh_dir, f"{int(time.time())}.obj"))