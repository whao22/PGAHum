import os
import cv2
import argparse
from pyhocon import ConfigFactory
import torch
import numpy as np
import time
from tqdm import tqdm
from glob import glob
import trimesh
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('.')
from libs import module_config
from libs.utils.geometry_utils import extract_geometry, remove_outliers, render_normal_from_mesh

# Arguments
parser = argparse.ArgumentParser(description='Extract Geometry from SDF Network.')
parser.add_argument('--conf', type=str, help='Path to config file.', default="confs/hfavatar-zjumocap/ZJUMOCAP-377-4gpus.conf")
parser.add_argument('--base_exp_dir', type=str, default="exp/CoreView_377_1709621919_slurm_mvs_1_1_1_true")
parser.add_argument('--n_frames', type=int, default=1, help="Number of frames to extract geometry from.")
parser.add_argument('--resolution', type=int, default=256)
parser.add_argument('--mcthreshold', type=float, default=0.0)
parser.add_argument('--render_normal', type=bool, default=True, help="Whether to render normal map or not.")
parser.add_argument('--mode', type=str, default='val', help="val / odp: (out-of-distribution pose)")
parser.add_argument('--device', type=str, default="cuda:3", help="cuda / cpu")

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
        else:
            gpu_data[key] = val

    return gpu_data
    
if  __name__ == '__main__':
    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.conf)
    out_dir = args.base_exp_dir
    resolution = args.resolution
    threshold = args.mcthreshold
    device = torch.device(args.device)
    
    conf['dataset']['test_views'] = [22]
    conf['dataset']['test_subsampling_rate'] = 100
    conf['dataset']['test_start_frame'] = 240
    conf['dataset']['test_end_frame'] = 250
    conf['dataset']['res_level'] = 1
    if args.mode == 'odp':
        conf['dataset']['dataset'] = 'aistplusplus_odp'
        conf['dataset']['novel_pose_folder'] = 'data/AIST++/motions/gBR_sBM_cAll_d04_mBR1_ch06.pkl'
    
    # Model
    print("Load model ...")
    hfavatar = module_config.get_model(conf).to(device)
    dataset = module_config.get_dataset('test', conf)

    # Load State Dict
    checkpoint_path = sorted(glob(os.path.join(out_dir, "checkpoints/epoch*.ckpt")))[-1]
    # checkpoint_path = os.path.join(out_dir, "checkpoints/last.ckpt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('No checkpoint is found!')
    print("Load state dict ...")
    hfavatar.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    
    smpl_sdf = np.load(os.path.join(conf.dataset.data_dir, conf.dataset.train_split[0], "smpl_sdf.npy"), allow_pickle=True).item()
    
    n_frames = args.n_frames
    mesh_dir = os.path.join(out_dir, 'odp' if args.mode == 'odp' else '', 'meshes')
    os.makedirs(mesh_dir, exist_ok=True)
    for frame in tqdm(range(n_frames)):
        data = dataset.gen_rays_for_infer(frame)
        data = cpu_data_to_gpu(data, args.device)
        dst_vertices = data['dst_vertices']
        # bound_min = dst_vertices.min(0)[0] - 0.1 # ndarray, [3]
        # bound_max = dst_vertices.max(0)[0] + 0.1 # ndarray, [3]
        bound_min = torch.ones_like(dst_vertices.min(0)[0]) * -1
        bound_max = torch.ones_like(dst_vertices.max(0)[0])
        
        dst_Rs, dst_Ts = pose_refine(hfavatar, data['dst_Rs'], data['dst_Ts'], data['dst_posevec'], hfavatar.total_bones)
        dst_gtfms = hfavatar.motion_basis_computer(dst_Rs, dst_Ts[None], data['cnl_gtfms'][None], data['tjoints'][None])
        dst_gtfms = torch.matmul(dst_gtfms, torch.inverse(data['gtfs_02v'])) # (B, 24, 4, 4)
        
        dst_vertices = np.load("data/body_models/misc/v_templates.npz")['male']
        dst_vertices = torch.from_numpy(dst_vertices).float().to(device)[None, ...]
        gtfs_02v = torch.inverse(data['gtfs_02v'])
        dst_vertices = backward_lbs(dst_vertices, gtfs_02v[None, ...], data['skinning_weights'][None, ...])
        
        sdf_kwargs = {
            "cnl_bbmin": smpl_sdf['bbmin'],
            "cnl_bbmax": smpl_sdf['bbmax'],
            "cnl_grid_sdf": torch.from_numpy(smpl_sdf['sdf_grid']).float().to(device),
            "dst_vertices": dst_vertices,
        }
        deform_kwargs = {
            "skinning_weights": data['skinning_weights'][None],
            "dst_gtfms": torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 24, 1, 1).to(dst_gtfms),
            "dst_posevec": torch.zeros_like(data['dst_posevec'][None]),
            "dst_vertices": data['dst_vertices'][None]
        }
        
        # Export .obj mesh
        print("Start extract geometry ...")
        vertices, triangles = extract_geometry(bound_min, bound_max, resolution, threshold, hfavatar, sdf_kwargs, deform_kwargs, device=device)
        
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh = remove_outliers(mesh)
        mesh.export(os.path.join(mesh_dir, f"{int(time.time())}_{frame}.obj"))
        vertices = mesh.vertices.astype(np.float32)
        triangles = mesh.faces.astype(np.int32)
        
        # Export .npy mesh data
        mesh_data = {
            'vertices': vertices,
            'triangles': triangles,
            'E': data['extrinsic'].cpu().numpy(),
            'K': data['intrinsic'].cpu().numpy(),
            'camera_e': data['camera_e'].cpu().numpy(),
            'rh': data['rh'].cpu().numpy(),
            'th': data['th'].cpu().numpy(),
            'hit_mask': data['hit_mask'].cpu().numpy(),
            'alpha_mask': data['batch_rays'][..., 9:10].cpu().numpy(),
            'H': data['height'],
            'W': data['width']
        }
        np.save(os.path.join(mesh_dir, f"{int(time.time())}_{frame}.npy"), mesh_data)
        
        if args.render_normal:
            # Export norml map
            vertices = torch.from_numpy(vertices).float().to(device)
            R_ext = torch.from_numpy(R.from_euler('xyz', [90, -120, -20], degrees=True).as_matrix()).float().to(device)
            vertices = torch.matmul(R_ext, vertices.reshape(-1, 3, 1))[..., 0]
            vertices = torch.matmul(data['rh'].T, vertices.reshape(-1, 3, 1))[..., 0]
            vertices = (vertices + data['th'].reshape(1, 3)).unsqueeze(0)
            triangles = torch.from_numpy(triangles.astype(np.int64)).long().to(device)
            triangles = triangles[..., [0,2,1]].unsqueeze(0)

            cam_rot = (data['camera_e'][..., :3, :3]).unsqueeze(0)
            cam_trans = (data['camera_e'][..., :3, 3]).unsqueeze(0)
            K = (data['intrinsic']).unsqueeze(0)
            H = data['height']
            W = data['width']
            mask = data['hit_mask'].reshape(H, W)
            
            normal_map = render_normal_from_mesh(vertices, triangles, cam_rot, cam_trans, K, mask, device, H, W)
            cv2.imwrite(os.path.join(mesh_dir, f"{int(time.time())}_{frame}_normal.png"), normal_map.cpu().numpy().astype(np.uint8))