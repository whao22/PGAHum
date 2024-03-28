import os
import argparse
from pyhocon import ConfigFactory
import torch
import numpy as np
import time
from glob import glob
import trimesh
from torch.utils.data import DataLoader
from libs import module_config
from libs.utils.geometry_utils import extract_geometry

# Arguments
parser = argparse.ArgumentParser(description='Extract Geometry from SDF Network.')
parser.add_argument('--conf', type=str, help='Path to config file.', default="confs/hfavatar-people_snapshot/PeopleSnapshot-male-3-casual-mono-4gpus.conf")
parser.add_argument('--base_exp_dir', type=str, default="exp/Peoplesnapshot-male-3-casual_1711424734_slurm_mono_1_1_3_true")
parser.add_argument('--frames', type=list, default=[3], help='List of frames to extract geometry.')
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--mcthreshold', type=float, default=0.0)
parser.add_argument('--device', type=str, default="cuda:1", help="cuda / cpu")

def multiply_corrected_Rs(Rs, correct_Rs, total_bones):
    total_bones = total_bones - 1
    return torch.matmul(Rs.reshape(-1, 3, 3),
                        correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)
        
def pose_refine(model, dst_Rs, dst_Ts, dst_posevec, total_bones):
    dst_Rs = dst_Rs.unsqueeze(0) # (B, 24, 3, 3)
    # forward pose refine
    pose_out = model.pose_decoder(dst_posevec)
    refined_Rs = pose_out['Rs'] # (B, 23, 3, 3) Rs matrix
    refined_Ts = pose_out.get('Ts', None)
    
    dst_Rs_no_root = dst_Rs[:, 1:, ...] # (1, 23, 3, 3)
    dst_Rs_no_root = multiply_corrected_Rs(dst_Rs_no_root, refined_Rs, total_bones) # (B, 23, 3, 3)
    dst_Rs = torch.cat([dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1) # (B, 24, 3, 3)
    
    # correct dst_Ts
    if refined_Ts is not None:
        dst_Ts = dst_Ts + refined_Ts

    return dst_Rs, dst_Ts, 

def cpu_data_to_gpu(cpu_data, device, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = []

    gpu_data = {}
    for key, val in cpu_data.items():
        if key in exclude_keys:
            continue

        if isinstance(val, list):
            assert len(val) > 0
            if not isinstance(val[0], str): # ignore string instance
                gpu_data[key] = [x.to(device) for x in val]
        elif isinstance(val, dict):
            gpu_data[key] = cpu_data_to_gpu(val, device, exclude_keys)
        elif isinstance(val, np.ndarray):
            gpu_data[key] = torch.from_numpy(val).float().to(device)

    return gpu_data
    
if  __name__ == '__main__':
    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.conf)
    out_dir = args.base_exp_dir
    resolution = args.resolution
    threshold = args.mcthreshold
    device = torch.device(args.device)
    
    # conf['dataset']['test_views'] = [44]
    # conf['dataset']['test_subsampling_rate'] = 100
    # conf['dataset']['test_start_frame'] = 260
    # conf['dataset']['test_end_frame'] = 300
    
    # Model
    print("Load model ...")
    hfavatar = module_config.get_model(conf).to(device)
    dataset = module_config.get_dataset('test', conf)

    # Load State Dict
    # checkpoint_path = sorted(glob(os.path.join(out_dir, "checkpoints/epoch*.ckpt")))[2]
    checkpoint_path = os.path.join(out_dir, "checkpoints/last.ckpt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('No checkpoint is found!')
    print("Load state dict ...")
    hfavatar.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    
    smpl_sdf = np.load(os.path.join(conf.dataset.data_dir, conf.dataset.train_split[0], "smpl_sdf.npy"), allow_pickle=True).item()
    
    frame_list = args.frames
    for frame in frame_list:
        data = dataset.gen_rays_for_infer(frame)
        data = cpu_data_to_gpu(data, args.device)
        dst_vertices = data['dst_vertices']
        bound_min = dst_vertices.min(0)[0] - 0.1 # ndarray, [3]
        bound_max = dst_vertices.max(0)[0] + 0.1 # ndarray, [3]
        
        dst_Rs, dst_Ts = pose_refine(hfavatar, data['dst_Rs'], data['dst_Ts'], data['dst_posevec'], hfavatar.total_bones)
        dst_gtfms = hfavatar.motion_basis_computer(dst_Rs, dst_Ts[None], data['cnl_gtfms'][None], data['tjoints'][None])
        dst_gtfms = torch.matmul(dst_gtfms, torch.inverse(data['gtfs_02v'])) # (B, 24, 4, 4)
        
        sdf_kwargs = {
            "cnl_bbmin": smpl_sdf['bbmin'],
            "cnl_bbmax": smpl_sdf['bbmax'],
            "cnl_grid_sdf": torch.from_numpy(smpl_sdf['sdf_grid']).float().to(device),
        }
        deform_kwargs = {
            "skinning_weights": data['skinning_weights'][None],
            "dst_gtfms": dst_gtfms,
            "dst_posevec": data['dst_posevec'][None],
            "dst_vertices": data['dst_vertices'][None]
        }
        
        print("Start extract geometry ...")
        vertices, triangles = extract_geometry(bound_min, bound_max, resolution, threshold, hfavatar, sdf_kwargs, deform_kwargs, device=device)
        
        mesh_dir = os.path.join(out_dir, 'meshes')
        os.makedirs(mesh_dir, exist_ok=True)
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(mesh_dir, f"{int(time.time())}_{frame}.obj"))