import sys
sys.path.append('.')

from libs.utils.general_utils import get_02v_bone_transforms
from scipy.spatial.transform import Rotation
from tools.smpl.smpl_numpy import SMPL
import mcubes
import mesh2sdf
import argparse
from glob import glob
from tqdm import tqdm
import cv2
import os.path as osp
import os
import yaml
import numpy as np
import pickle
import h5py
import smplx
import torch


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    cfg = parser.parse_args()

    with open(cfg.conf, 'r') as f:
        cfg = yaml.load(f)
    return cfg


def prepare_smpl_sdf(vertices, volume_size):
    PADDING = 0.05
    faces = np.load('data/body_models/misc/faces.npz')['faces']

    maxv = abs(vertices).max()
    bbmax = maxv + PADDING
    bbmin = -maxv - PADDING

    vertices_norm = (vertices - bbmin) / (bbmax - bbmin + 1e-10)
    vertices_norm = (vertices_norm - 0.5) * 2
    sdf_grid = mesh2sdf.compute(vertices_norm, faces, size=volume_size)

    smpl_sdf = {
        "sdf_grid": sdf_grid,
        "bbmax": bbmax,
        "bbmin": bbmin,
    }
    return smpl_sdf


def process_view(root, view, sid, save_root, end_frame, annots, scale_ratio):
    print(f"Process images and maskes for view {view} ... ")
    # masks and images
    mask_file_list = sorted(glob(osp.join(root, "mask", f"{view}".zfill(2), "*.png")))
    images_file_list = sorted(glob(osp.join(root, "images", f"{view}".zfill(2), "*.jpg")))
    frame_num = len(mask_file_list)
    if end_frame >= 0 and end_frame < frame_num:
        frame_num = end_frame
    save_mask_path = osp.join(save_root, 'masks')
    os.makedirs(save_mask_path, exist_ok=True)
    save_images_path = osp.join(save_root, 'images')
    os.makedirs(save_images_path, exist_ok=True)
    
    for ind in tqdm(range(frame_num), desc='images and masks'):
        mask_file = mask_file_list[ind]
        image_file = images_file_list[ind]
        _, frame_id = osp.split(image_file)
        _, frame_id = osp.split(mask_file)
        
        mask_from_file = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE).astype(np.bool_)
        image = cv2.imread(image_file)
        mask_from_image = image.sum(-1) > 10
        mask = np.logical_and(mask_from_file, mask_from_image)

        cv2.imwrite(osp.join(save_mask_path, frame_id), mask*255)
        cv2.imwrite(osp.join(save_images_path, frame_id[:-4]+".png"), image)

    # load cameras
    cams = annots['cams']
    cam_Ks = np.array(cams['K'])[view].astype('float32')
    cam_Rs = np.array(cams['R'])[view].astype('float32')
    cam_Ts = np.array(cams['T'])[view].astype('float32') / 1000.
    cam_Ds = np.array(cams['D'])[view].astype('float32')
    
    cam_Ks[:2, :2] = cam_Ks[:2, :2] * scale_ratio
    K = cam_Ks  # (3, 3)
    D = cam_Ds
    E = np.eye(4)  # (4, 4)
    cam_T = cam_Ts[:3, 0] * scale_ratio
    E[:3, :3] = cam_Rs
    E[:3, 3] = cam_T
    
    cameras = {} 
    for ind in tqdm(range(frame_num), desc='cameras'):
        out_name = f'{ind}'.zfill(6)
        cameras[out_name] = {
            'intrinsics': K,
            'extrinsics': E,
            'distortions': D
        }
    
    # write camera infos
    with open(os.path.join(save_root, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cameras, f)

    return frame_num
    
def main(cfg):
    root = osp.join(cfg['dataset']['data_path'], cfg['dataset']['subject'])
    sid = cfg['dataset']['start_frame']
    end_frame = cfg['dataset']['end_frame']
    model_path = cfg['dataset']['model_path']
    gender = cfg['dataset']['sex']
    scale_ratio = cfg['dataset']['scale_ratio']
    save_root = osp.join(cfg['output']['dir'], cfg['output']['name'])
    select_views = cfg['dataset']['preprocess_views']
    # select_views = [1]
    volume_size = cfg['dataset']['volume_size']
    skinning_weights = dict(
        np.load('data/body_models/misc/skinning_weights_all.npz'))[gender]
    annots = np.load(os.path.join(root, 'annots.npy'), allow_pickle=True).item()
    
    frame_num = 1300
    # for view in select_views:
    #     view_dir = osp.join(save_root, f"{view}")
    #     os.makedirs(view_dir, exist_ok=True)
    #     frame_num = process_view(root, view, sid, view_dir, end_frame, annots, scale_ratio)

    # load motion
    motion = np.load(osp.join(root,'motion.npz'))
    poses = motion['poses'] # (frame_num, 156)
    shapes = motion['shapes'] # (frame_num, 16)
    Rhs = motion['Rh'] # (frame_num, 3)
    Ths = motion['Th'] # (frame_num, 3)
    
    mesh_infos = {}
    if cfg['dataset']['subject'] == 'manuel':
        smplx_model = smplx.create(model_path=model_path, 
                                   model_type='smplx', 
                                   gender=gender, 
                                   num_betas=shapes.shape[-1],
                                   num_pca_comps=15*3).cuda()
    else:
        smpl_model = SMPL(sex=gender, model_dir=model_path)
    
    for ind in tqdm(range(frame_num), desc='mesh_infos'):
        Rh = Rhs[ind]
        Th = Ths[ind]
        betas = shapes[ind]
        thetas = poses[ind]
        if cfg['dataset']['subject'] == 'manuel':
            
            betas = torch.from_numpy(betas).unsqueeze(0).float().cuda()
            thetas = torch.from_numpy(thetas).unsqueeze(0).float().cuda()
            
            output = smplx_model(
                        betas=betas, 
                        global_orient=thetas[:, :3], 
                        body_pose=thetas[:, 3:66], 
                        left_hand_pose=thetas[:, 111:156],
                        right_hand_pose=thetas[:, 66:111],
                        return_verts=True,
                        return_full_pose=True)
            posed_vertices = output.vertices.detach().cpu().numpy().squeeze()
            joints = output.joints.detach().cpu().numpy().squeeze()
        
        else:
            betas = shapes[ind][:10]
            thetas = poses[ind][:72]
            posed_vertices, joints = smpl_model(thetas, betas)
            
        mesh_infos[f'{str(ind).zfill(6)}'] = {
            'Rh': Rh,
            'Th': Th,
            'poses': thetas,
            'beats': betas,
            'posed_vertices': posed_vertices * scale_ratio,
            'joints': joints * scale_ratio,
        }

        # dilate mesh
        sdf_resolution = 64
        delta_sdf = 0.03
        posed_sdf_dict = prepare_smpl_sdf(posed_vertices, sdf_resolution)
        posed_sdf_grid = posed_sdf_dict['sdf_grid'] - delta_sdf
        posed_bbmax = posed_sdf_dict['bbmax']
        posed_bbmin = posed_sdf_dict['bbmin']

        dilated_vertices, dilated_triangles = mcubes.marching_cubes(
            posed_sdf_grid, 0.)
        dilated_vertices = dilated_vertices / sdf_resolution * \
            (posed_bbmax - posed_bbmin) + posed_bbmin

        mesh_infos[f'{str(ind).zfill(6)}'].update({
            'dilated_triangles': dilated_triangles,
            'dilated_vertices': dilated_vertices * scale_ratio,
        })

    # write mesh infos
    with open(os.path.join(save_root, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)

    # write canonical joints
    shape = shapes.mean(axis=0)
    if cfg['dataset']['subject'] == 'manuel':
        output = smplx_model(betas=torch.from_numpy(shape).unsqueeze(0).float().cuda(), 
                             global_orient=torch.zeros(1, 3).float().cuda(), 
                             body_pose=torch.zeros(1, 63).float().cuda(), 
                             left_hand_pose=torch.zeros(1, 33).float().cuda(),
                             right_hand_pose=torch.zeros(1, 33).float().cuda(),
                             return_verts=True,
                             return_full_pose=True)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        template_joints = output.joints.detach().cpu().numpy().squeeze()
    else:
        vertices, template_joints = smpl_model(np.zeros(72), shape)
    vertices = vertices * scale_ratio
    template_joints = template_joints * scale_ratio

    with open(os.path.join(save_root, 'canonical_joints.pkl'), 'wb') as f:
        pickle.dump({
            'joints': template_joints,
            'vertices': vertices
        }, f)

    # process init sdf
    smpl_sdf_path = os.path.join(save_root, 'smpl_sdf.npy')
    if cfg['dataset']['use_smpl_sdf'] and not os.path.exists(smpl_sdf_path):
        print("Process SMPL prior sdf ... ")
        gtfs_02v = get_02v_bone_transforms(template_joints)
        T = np.matmul(skinning_weights, gtfs_02v.reshape(
            [-1, 16])).reshape([-1, 4, 4])
        vertices_v = np.matmul(
            T[:, :3, :3], vertices[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]

        # prepare smpl sdf
        smpl_sdf = prepare_smpl_sdf(vertices_v, volume_size)
        np.save(smpl_sdf_path, smpl_sdf)

if __name__ == '__main__':
    ''' python tools/prepare_synthetic_human/prepare_synthetic_human.py --conf tools/prepare_synthetic_human/confs_preprocess/jody.yaml '''
    cfg = parse_cfg()
    main(cfg)
