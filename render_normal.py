import cv2
import torch
import numpy as np
from glob import glob
from scipy.spatial.transform import Rotation as R
from libs.utils.geometry_utils import render_normal_from_mesh

if __name__ == '__main__':
    device = torch.device('cuda:3')

    nuy_file = sorted(glob("exp/CoreView_377_1709621919_slurm_mvs_1_1_1_true/odp/meshes/*.npy"))[-1]
    mesh_data = np.load(nuy_file, allow_pickle=True).item()

    # extrinsic matrix
    extrinsic = torch.from_numpy(mesh_data['camera_e']).unsqueeze(0).to(device)
    cam_rot = extrinsic[..., :3,:3]
    cam_trans = extrinsic[..., :3,3] * torch.tensor([1, 1, 1], device=device).reshape(1,3)
    
    # intrinsic matrix
    K = mesh_data['K']
    K = torch.tensor(K, device=device).reshape(1,3,3)
    # K[:2,:2] *= 2
    
    # rotation and translation for human
    Rh = mesh_data['rh']
    Th = mesh_data['th'] # / 93 + np.array([0.3,-1.5,-1.2])
    # mesh = trimesh.load("test/eval_chd/data/377_pet-neus_gt.ply")
    # verts = mesh.vertices
    # faces = mesh.faces.astype(np.int64)
    verts = mesh_data['vertices']
    # verts = np.matmul(R.from_euler('xyz', [90, -120, -20], degrees=True).as_matrix(), verts.reshape(-1, 3, 1))[..., 0]
    verts = np.matmul(Rh.T, verts.reshape(-1, 3, 1))[..., 0]
    verts = torch.from_numpy(verts + Th.reshape(1, 3)).float().to(device).unsqueeze(0)
    faces = mesh_data['triangles'].astype(np.int64)
    faces = torch.from_numpy(faces[..., [0,2,1]]).to(device).unsqueeze(0)
    H = mesh_data['H']
    W = mesh_data['W']
    # mask = torch.from_numpy(mesh_data['hit_mask'].reshape(H, W)).to(device)
    mask = torch.from_numpy(mesh_data['alpha_mask'].reshape(H, W)).to(device)
    
    normal_image = render_normal_from_mesh(verts, faces, cam_rot, cam_trans, K, mask, device, H, W,)
    cv2.imwrite("normal_image.png", normal_image.cpu().numpy().astype(np.uint8))