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



def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    cfg = parser.parse_args()

    with open(cfg.conf, 'r') as f:
        cfg = yaml.load(f)
    return cfg


def prepare_smpl_sdf(vertices, volume_size):
    PADDING = 0.3
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


def process_view(root, start_frame, end_frame, save_root):
    masks_dir = osp.join(root,'masks')
    images_dir = osp.join(root, 'imgs')
    normals_dir = osp.join(root, 'normals')
    frame_num = len(glob(osp.join(masks_dir, '*.png')))
    
    # masks
    mask_root = osp.join(save_root, 'masks')
    os.makedirs(mask_root, exist_ok=True)
    for ind in tqdm(range(start_frame, frame_num), desc='masks'):
        mask = cv2.imread(osp.join(masks_dir, '%06d.png' % ind))
        cv2.imwrite(osp.join(mask_root, 'frame_%06d.png' % ind), mask)

    # images
    rgb_root = osp.join(save_root, 'images')
    os.makedirs(rgb_root, exist_ok=True)
    for ind in tqdm(range(start_frame, frame_num), desc='images'):
        image = cv2.imread(osp.join(images_dir, '%06d.png' % ind))
        cv2.imwrite(osp.join(rgb_root, 'frame_%06d.png' % ind), image)
        
    # normals
    normals_root = osp.join(save_root, 'normals')
    os.makedirs(normals_root, exist_ok=True)
    for ind in tqdm(range(start_frame, frame_num), desc='normals'):
        normal = cv2.imread(osp.join(normals_dir, '%06d.png' % ind))
        cv2.imwrite(osp.join(normals_root, 'frame_%06d.png' % ind), normal)

    # cameras
    camera_data = np.load(osp.join(root, 'camera.npz'), allow_pickle=True)
    fx = camera_data['fx']
    fy = camera_data['fy']
    cx = camera_data['cx']
    cy = camera_data['cy']
    quat_wxyz = camera_data['quat']
    quat_xyzw = np.roll(quat_wxyz, -1)
    # rt_mat = Rotation.from_quat(quat_xyzw).as_matrix()
    rt_mat = np.eye(3)
    trans = camera_data['T']
    
    cameras_info = {}
    for ind in tqdm(range(frame_num), desc='cameras'):
        E = np.eye(4)
        E[:3, :3] = rt_mat
        E[:3, 3] = trans
        D = None
        camera_info = {
            'intrinsics': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]).astype(np.float32),
            'extrinsics': E.astype(np.float32),
            'distortions': D
        }
        cameras_info[f'frame_{str(ind).zfill(6)}'] = camera_info

    # write camera infos
    with open(os.path.join(save_root, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cameras_info, f)

    return frame_num

def main(cfg):
    root = osp.join(cfg['dataset']['data_path'], cfg['dataset']['subject'])
    start_frame = cfg['dataset']['start_frame']
    end_frame = cfg['dataset']['end_frame']
    model_path = cfg['dataset']['model_path']
    gender = cfg['dataset']['sex']
    scale_ratio = cfg['dataset']['scale_ratio']
    save_root = osp.join(cfg['output']['dir'], cfg['output']['name'])
    select_views = cfg['dataset']['preprocess_views']
    volume_size = cfg['dataset']['volume_size']
    skinning_weights = dict(
        np.load('data/body_models/misc/skinning_weights_all.npz'))[gender]

    frame_num = None
    for view in select_views:
        images_dir = osp.join(save_root, f"{view}")
        os.makedirs(images_dir, exist_ok=True)
        frame_num = process_view(root, start_frame, end_frame, images_dir)

    # mesh info
    smpl_rec_data = np.load(osp.join(root,'smpl_rec.npz'), allow_pickle=True)
    shape = smpl_rec_data['shape'] # (10,)
    poses = smpl_rec_data['poses'] # (frame_num, 24, 3)
    trans = smpl_rec_data['trans'] # (frame_num, 3)
    
    mesh_infos = {}
    smpl_model = SMPL(sex=gender, model_dir=model_path)
    for ind in tqdm(range(frame_num), desc='mesh_infos'):
        Rh = poses[ind][0].copy()
        Th = trans[ind].copy()
        betas = shape
        thetas = poses[ind].reshape(-1).copy()
        thetas[:3] = 0.
        posed_vertices, joints = smpl_model(thetas, betas)
        mesh_infos[f'frame_{str(ind).zfill(6)}'] = {
            'Rh': Rh,
            'Th': Th,
            'poses': thetas,
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

        dilated_vertices, dilated_triangles = mcubes.marching_cubes(
            posed_sdf_grid, 0.)
        dilated_vertices = dilated_vertices / sdf_resolution * \
            (posed_bbmax - posed_bbmin) + posed_bbmin

        mesh_infos[f'frame_{str(ind).zfill(6)}'].update({
            'dilated_triangles': dilated_triangles,
            'dilated_vertices': dilated_vertices * scale_ratio,
        })

    # write mesh infos
    with open(os.path.join(save_root, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)

    # write canonical joints
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
    ''' python tools/prepare_selfrecon_synthesis/people_synthesis_process.py --conf tools/prepare_selfrecon_synthesis/confs_preprocess/male-outfit2.yaml '''
    cfg = parse_cfg()
    main(cfg)
