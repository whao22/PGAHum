import os
import sys
sys.path.append('.')

import yaml
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import mesh2sdf
import trimesh

from libs.utils.images_utils import load_image, save_image, to_3ch_image
from libs.utils.file_utils import split_path
from libs.utils.general_utils import get_02v_bone_transforms
from tools.smpl.smpl_numpy import SMPL

def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask', img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk_path = os.path.join(subject_dir, 'mask_cihp', img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))[:, :, 0]
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8)
    msk[msk == 1] = 255

    return msk


def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        cfg = yaml.load(f)
    return cfg

def process_view(cfg, subject, select_view, subject_dir, smpl_params_dir, annots, smpl_model, skinning_weights):
    start_frame = cfg['dataset']['start_frame']
    end_frame = cfg['dataset']['end_frame']
    scale_ratio = cfg['dataset']['scale_ratio']
    volume_size = cfg['dataset']['volume_size']
    
    
    # load cameras
    cams = annots['cams']
    cam_Ks = np.array(cams['K'])[select_view].astype('float32')
    cam_Rs = np.array(cams['R'])[select_view].astype('float32')
    cam_Ts = np.array(cams['T'])[select_view].astype('float32') / 1000.
    cam_Ds = np.array(cams['D'])[select_view].astype('float32')
    
    cam_Ks[:2, :2] = cam_Ks[:2, :2]
    K = cam_Ks  # (3, 3)
    D = cam_Ds[:, 0]
    E = np.eye(4)  # (4, 4)
    cam_T = cam_Ts[:3, 0] * scale_ratio
    E[:3, :3] = cam_Rs
    E[:3, 3] = cam_T

    # load image paths
    img_path_frames_views = annots['ims']
    img_paths = np.array([
        np.array(multi_view_paths['ims'])[select_view]
        for multi_view_paths in img_path_frames_views
    ])
    img_paths = img_paths[start_frame:end_frame]
    
    # prepare out dir
    output_path = os.path.join(cfg['output']['dir'], subject if 'name' not in cfg['output'].keys() else cfg['output']['name'], str(select_view))
    os.makedirs(output_path, exist_ok=True)
    out_img_dir = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')

    cameras = {}
    mesh_infos = {}
    all_betas = []
    for idx, ipath in enumerate(tqdm(img_paths)):
        out_name = 'frame_{:06d}'.format(idx)
        img_path = os.path.join(subject_dir, ipath)
        # load image
        img = np.array(load_image(img_path))

        if subject in ['313', '315']:
            _, image_basename, _ = split_path(img_path)
            start = image_basename.find(')_')
            smpl_idx = int(image_basename[start+2: start+6])
        else:
            smpl_idx = idx

        # load smpl parameters
        smpl_params = np.load(
            os.path.join(smpl_params_dir, f"{smpl_idx}.npy"),
            allow_pickle=True).item()

        betas = smpl_params['shapes'][0]  # (10,)
        poses = smpl_params['poses'][0]  # (72,)
        Rh = smpl_params['Rh'][0]  # (3,)
        Th = smpl_params['Th'][0] * scale_ratio  # (3,)

        all_betas.append(betas)

        # write camera info
        cameras[out_name] = {
            'intrinsics': K,
            'extrinsics': E,
            'distortions': D
        }

        # write mesh info
        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        posed_vertices, joints = smpl_model(poses, betas)
        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th ,
            'poses': poses,
            'beats': betas,
            'posed_vertices': posed_vertices * scale_ratio,
            'joints': joints * scale_ratio,
            'tpose_joints': tpose_joints * scale_ratio
        }

        # load and write mask
        mask = get_mask(subject_dir, ipath)
        save_image(to_3ch_image(mask),
                   os.path.join(out_mask_dir, out_name+'.png'))

        # write image
        out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
        save_image(img, out_image_path)

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    vertices, template_joints = smpl_model(np.zeros(72), avg_betas) 
    vertices = vertices * scale_ratio
    template_joints = template_joints * scale_ratio
    
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:
        pickle.dump({
            'joints': template_joints, 
            'vertices': vertices 
            }, f)

    if cfg['dataset']['use_smpl_sdf']:
        gtfs_02v = get_02v_bone_transforms(template_joints)
        T = np.matmul(skinning_weights, gtfs_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices_v = np.matmul(T[:, :3, :3], vertices[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]
        
        # prepare smpl sdf
        prepare_smpl_sdf(output_path, vertices_v, volume_size)

def prepare_smpl_sdf(output_path, vertices, volume_size):
    PADDING = 0.05
    faces = np.load('data/body_models/misc/faces.npz')['faces']

    maxv = abs(vertices).max()
    bbmax = maxv + PADDING
    bbmin = -maxv - PADDING

    vertices_norm = (vertices - bbmin) / (bbmax - bbmin + 1e-10)
    vertices_norm = (vertices_norm - 0.5) * 2
    sdf_grid = mesh2sdf.compute(vertices_norm, faces, size=volume_size)

    smpl_sdf={
        "sdf_grid": sdf_grid,
        "bbmax": bbmax,
        "bbmin": bbmin,
    }
    np.save(os.path.join(output_path, 'smpl_sdf.npy'), smpl_sdf)

def main(cfg):
    subject = cfg['dataset']['subject']
    dataset_dir = cfg['dataset']['zju_mocap_path']
    select_views = cfg['dataset']['preprocess_views']
    sex = cfg['dataset']['sex']
    model_path = cfg['dataset']['model_path']
    
    # obtain directories
    subject_dir = os.path.join(dataset_dir, f"CoreView_{subject}")
    smpl_params_dir = os.path.join(subject_dir, "new_params")
    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()
    smpl_model = SMPL(sex=sex, model_dir=model_path)
    skinning_weights = dict(np.load('data/body_models/misc/skinning_weights_all.npz'))[sex]
    
    # process views
    for view in select_views:
        process_view(cfg, subject, view, subject_dir, smpl_params_dir, annots, smpl_model, skinning_weights)
        
    
if __name__ == '__main__':
    '''python tools/prepare_zju_mocap/prepare_dataset.py --conf tools/prepare_zju_mocap/confs_preprocess/377.yaml'''
    cfg = parse_cfg()
    main(cfg)
