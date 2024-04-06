import numpy as np
import torch
from glob import glob
import trimesh
from scipy.spatial.transform import Rotation as R
from chamferdist import ChamferDistance
from pytorch3d.loss import chamfer_distance

align_with_gt = False


def chd_hfavatar(exp_name, case):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_mesh_path = sorted(glob(f"exp/{exp_name}/meshes/*.obj"))[-1]
    targ_mesh_path = f"test/eval_chd/data/{case}_pet-neus_gt.ply"
    pred_mesh = trimesh.load_mesh(pred_mesh_path)
    pred_mesh_verts = trimesh.sample.sample_surface(pred_mesh, 10000)[0]
    targ_mesh = trimesh.load_mesh(targ_mesh_path)
    targ_mesh_verts = trimesh.sample.sample_surface(targ_mesh, 10000)[0]

    # warp source cloud
    source_cloud = torch.from_numpy(pred_mesh_verts)[None].float().to(device) # [1, N, 3]
    if not align_with_gt:
        npy_file = sorted(glob(f"exp/{exp_name}/meshes/*.npy"))[-1]
        mesh_data = np.load(npy_file, allow_pickle=True).item()
        rh = mesh_data['rh']
        th = mesh_data['th']
        rh = torch.from_numpy(rh).float().to(device).T @ torch.from_numpy(R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()).to(device).float()
        source_cloud = torch.matmul(rh, source_cloud.reshape(-1, 3, 1))[..., 0].reshape(1, -1, 3)
        source_cloud += torch.from_numpy(th).float().to(device).reshape(1, 1, 3)
        # trimesh.Trimesh(source_cloud.reshape(-1,3).cpu().numpy()).export("aaa.ply")
    target_cloud = torch.from_numpy(targ_mesh_verts)[None].float().to(device) # [1, M, 3]
    cd = chamfer_distance(source_cloud, target_cloud)
    return cd

def chd_arah(case):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_mesh_path = f"test/eval_chd/data/{case}_arah.{'ply' if case == '377' else 'obj'}"
    targ_mesh_path = f"test/eval_chd/data/{case}_pet-neus_gt.ply"

    pred_mesh = trimesh.load_mesh(pred_mesh_path)
    pred_mesh_verts = trimesh.sample.sample_surface(pred_mesh, 1000)[0]
    targ_mesh = trimesh.load_mesh(targ_mesh_path)
    targ_mesh_verts = trimesh.sample.sample_surface(targ_mesh, 1000)[0]

    source_cloud = torch.from_numpy(pred_mesh_verts)[None].float().to(device) # [1, N, 3]
    target_cloud = torch.from_numpy(targ_mesh_verts)[None].float().to(device) # [1, M, 3]
    cd = chamfer_distance(source_cloud, target_cloud)
    return cd

def chd_nb(case):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_mesh_path = f"test/eval_chd/data/{case}_nb.ply"
    targ_mesh_path = f"test/eval_chd/data/{case}_pet-neus_gt.ply"

    pred_mesh = trimesh.load_mesh(pred_mesh_path)
    pred_mesh_verts = trimesh.sample.sample_surface(pred_mesh, 1000)[0] / 200
    targ_mesh = trimesh.load_mesh(targ_mesh_path)
    targ_mesh_verts = trimesh.sample.sample_surface(targ_mesh, 1000)[0]

    source_cloud = torch.from_numpy(pred_mesh_verts)[None].float().to(device) # [1, N, 3]
    target_cloud = torch.from_numpy(targ_mesh_verts)[None].float().to(device) # [1, M, 3]
    cd = chamfer_distance(source_cloud, target_cloud)
    return cd

if __name__ == '__main__':
    exp_name = 'CoreView_386_1711290959_slurm_mvs_1_1_3'
    cd = chd_hfavatar(exp_name, case='386')
    print("hfavatar: ", cd)
    cd = chd_arah(case='386')
    print("arah: ", cd)
    cd = chd_nb(case='386')
    print("nb: ", cd)
