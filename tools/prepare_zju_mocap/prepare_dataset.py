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
import mcubes
from libs.utils.images_utils import load_image, save_image, to_3ch_image
from libs.utils.file_utils import split_path
from libs.utils.general_utils import get_02v_bone_transforms
from tools.smpl.smpl_numpy import SMPL
from libs.utils.geometry_utils import extract_geometry

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

def process_view(subject_dir, img_paths, select_view, annots, output_path, scale_ratio=1.0):
    print(f"Process images and maskes for view {select_view} ... ")
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
    
    # prepare out dir
    out_img_dir = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')

    cameras = {}
    for idx, ipath in enumerate(tqdm(img_paths)):
        out_name = 'frame_{:06d}'.format(idx)
        img_path = os.path.join(subject_dir, ipath)
        # load image
        img = np.array(load_image(img_path))

        # write camera info
        cameras[out_name] = {
            'intrinsics': K,
            'extrinsics': E,
            'distortions': D
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


def prepare_smpl_sdf(vertices, volume_size):
    PADDING = 0.1
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
    return smpl_sdf

def main(cfg):
    subject = cfg['dataset']['subject']
    dataset_dir = cfg['dataset']['zju_mocap_path']
    select_views = cfg['dataset']['preprocess_views']
    sex = cfg['dataset']['sex']
    model_path = cfg['dataset']['model_path']
    start_frame = cfg['dataset']['start_frame']
    end_frame = cfg['dataset']['end_frame']
    scale_ratio = cfg['dataset']['scale_ratio']
    volume_size = cfg['dataset']['volume_size']
    
    # obtain directories
    subject_dir = os.path.join(dataset_dir, f"CoreView_{subject}")
    smpl_params_dir = os.path.join(subject_dir, "new_params")
    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()
    smpl_model = SMPL(sex=sex, model_dir=model_path)
    skinning_weights = dict(np.load('data/body_models/misc/skinning_weights_all.npz'))[sex]
    img_path_frames_views = annots['ims']
    
    # makedir output
    output_path = os.path.join(cfg['output']['dir'], subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)
    
    # process views
    for view in select_views:
        img_paths = np.array([
            np.array(multi_view_paths['ims'])[view]
            for multi_view_paths in img_path_frames_views
        ])
        img_paths = img_paths[start_frame:end_frame]
        
        output_path_view = os.path.join(output_path, str(view))
        os.makedirs(output_path_view, exist_ok=True)
        process_view(subject_dir, img_paths, view, annots, output_path_view)
    
    # process mesh info
    mesh_infos = {}
    all_betas = []
    print("Process Mesh info ... ")
    for idx, ipath in enumerate(tqdm(img_paths)):
        out_name = 'frame_{:06d}'.format(idx)
        img_path = os.path.join(subject_dir, ipath)

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

        # write mesh info
        posed_vertices, joints = smpl_model(poses, betas)
        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'beats': betas,
            'posed_vertices': posed_vertices * scale_ratio, 
            'joints': joints * scale_ratio,
        }
        
        # dilate mesh
        sdf_resolution = 64
        delta_sdf = 0.06
        posed_sdf_dict = prepare_smpl_sdf(posed_vertices, sdf_resolution)
        posed_sdf_grid = posed_sdf_dict['sdf_grid'] - delta_sdf
        posed_bbmax = posed_sdf_dict['bbmax']
        posed_bbmin = posed_sdf_dict['bbmin']
        
        dilated_vertices, dilated_triangles = mcubes.marching_cubes(posed_sdf_grid, 0.)    
        dilated_vertices = dilated_vertices / sdf_resolution * (posed_bbmax - posed_bbmin) + posed_bbmin

        mesh_infos[out_name].update({
            'dilated_triangles': dilated_triangles,
            'dilated_vertices': dilated_vertices * scale_ratio,
        })
    
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

    # process init sdf
    smpl_sdf_path = os.path.join(output_path, 'smpl_sdf.npy')
    if cfg['dataset']['use_smpl_sdf'] and not os.path.exists(smpl_sdf_path):
        print("Process SMPL prior sdf ... ")
        gtfs_02v = get_02v_bone_transforms(template_joints)
        T = np.matmul(skinning_weights, gtfs_02v.reshape([-1, 16])).reshape([-1, 4, 4])
        vertices_v = np.matmul(T[:, :3, :3], vertices[..., np.newaxis]).squeeze(-1) + T[:, :3, -1]
        
        # prepare smpl sdf
        smpl_sdf = prepare_smpl_sdf(vertices_v, volume_size)
        np.save(smpl_sdf_path, smpl_sdf)
    
if __name__ == '__main__':
    '''python tools/prepare_zju_mocap/prepare_dataset.py --conf tools/prepare_zju_mocap/confs_preprocess/377.yaml'''
    cfg = parse_cfg()
    main(cfg)
