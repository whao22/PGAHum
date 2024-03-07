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


def process_view(root, sid, save_root, end_frame):
    # masks
    frame_num = None
    with h5py.File(osp.join(root, 'masks.hdf5'), 'r') as ff:
        frame_num = ff['masks'].shape[0]
        if end_frame >= 0 and end_frame < frame_num:
            frame_num = end_frame
        assert frame_num > sid
        mask_root = osp.join(save_root, 'masks')
        os.makedirs(mask_root, exist_ok=True)
        for ind in tqdm(range(sid, frame_num), desc='masks'):
            cv2.imwrite(osp.join(mask_root, 'frame_%06d.png' %
                        (ind-sid)), ff['masks'][ind]*255)

    video = glob(osp.join(root, '*.mp4'))
    assert (len(video) == 1)
    video = video[0]

    cap = cv2.VideoCapture(video)
    rgb_root = osp.join(save_root, 'images')
    if not osp.isdir(rgb_root):
        os.makedirs(rgb_root)

    # images
    for ind in tqdm(range(frame_num), desc='rgbs'):
        check, img = cap.read()
        if ind < sid:
            continue
        if img.shape[:2] == (1080, 1920):
            img = img.transpose(1, 0, 2)
            img = img[:, ::-1, :]
        if not check:
            break
        cv2.imwrite(osp.join(rgb_root, 'frame_%06d.png' % (ind-sid)), img)
    cap.release()

    # cameras
    with open(osp.join(root, 'camera.pkl'), 'rb') as ff:
        cam_data = pickle.load(ff, encoding='latin1')
        center = cam_data['camera_c']
        focus = cam_data['camera_f']
        trans = cam_data['camera_t']
        rt = cam_data['camera_rt']
        distortions = cam_data['camera_k']
        # The cameras of snapshot dataset seems no rotation and translation
        assert (np.linalg.norm(rt) < 0.0001)

        rt_mat = cv2.Rodrigues(rt)[0] 
        rt_mat = rt_mat @ Rotation.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
        fx = focus[0]
        fy = focus[1]
        cx = center[0]
        cy = center[1]

    cameras_info = {}
    for ind in tqdm(range(frame_num), desc='cameras'):
        E = np.eye(4)
        E[:3, :3] = rt_mat
        E[:3, 3] = trans
        D = distortions
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
    sid = cfg['dataset']['start_frame']
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
        frame_num = process_view(root, sid, images_dir, end_frame)

    with h5py.File(osp.join(root, 'reconstructed_poses.hdf5'), 'r') as ff:
        shape = ff['betas'][:].reshape(10)
        poses = ff['pose'][:].reshape(-1, 24, 3)[sid:, :, :]
        trans = ff['trans'][:].reshape(-1, 3)[sid:, :]
        assert (poses.shape[0] >= frame_num -
                sid and trans.shape[0] >= frame_num-sid)
        # np.savez(osp.join(save_root,'smpl_rec.npz'),poses=poses,shape=shape,trans=trans,gender=gender)

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
        delta_sdf = 0.03
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
    ''' python tools/prepare_people_snapeshot/people_snapshot_process.py --conf tools/prepare_people_snapeshot/confs_preprocess/male-4-casual.yaml '''
    cfg = parse_cfg()
    main(cfg)
