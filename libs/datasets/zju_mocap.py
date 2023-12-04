import os
import glob
import cv2
import numpy as np
import json
# import logging
import trimesh
import numbers
import pyrender
import igl
from tqdm import tqdm
from torch.utils import data
from scipy.spatial.transform import Rotation

from libs.utils.general_utils import get_bound_2d_mask, get_near_far, get_02v_bone_transforms, rays_mesh_intersections_pcu

DEBUG = False

class ZJUMOCAPDataset(data.Dataset):
    ''' ZJU MoCap dataset class.
    '''

    def __init__(self, 
                 dataset_folder='',
                 subjects=['CoreView_313'],
                 mode='train',
                 img_size=(512, 512),
                 num_fg_samples=1024,
                 num_bg_samples=1024,
                 sampling_rate=1,
                 start_frame=0,
                 end_frame=-1,
                 views=[],
                 box_margin=0.05,
                 sampling='default',
                 erode_mask=True,):
        ''' Initialization of the the ZJU-MoCap dataset.

        Args:
            dataset_folder (str): dataset folder

            subjects (list of strs): which subjects to use
            mode (str): mode of the dataset. Can be either 'train', 'val' or 'test'
            img_size (int or tuple of ints): target image size we want to sample frome
            num_fg_samples (int): number of points to sample from foreground
            num_bg_samples (int): number of points to sample from background
            sampling_rate (int): sampling rate for video frames
            start_frame (int): start frame of the video
            end_frame (int): end frame of the video
            views (list of strs): which views to use
            box_margin (float): bounding box margin added to SMPL bounding box. This bounding box is used to determine sampling region in an image
            sampling (str): ray-sampling method. For current version of code, only 'default' is throughly tested
            erode_mask (bool): whether to erode ground-truth foreground masks, such that boundary pixels of masks are ignored
        '''
        assert (len(subjects) == 1) # TODO: we only support per-subject training at this point
        
        # flags
        self.mode = mode
        self.dataset_folder = dataset_folder
        self.num_fg_samples = num_fg_samples
        self.num_bg_samples = num_bg_samples
        self.sampling = sampling
        self.erode_mask = erode_mask
        self.box_margin = box_margin
        if isinstance(img_size, numbers.Number):
            self.img_size = (int(img_size), int(img_size))
        else:
            self.img_size = img_size
        self.homo_2d = self.init_grid_homo_2d(img_size[0], img_size[1])
        
        # load meta data
        self.faces = np.load('data/body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('data/body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('data/body_models/misc/posedirs_all.npz'))
        self.J_regressor = dict(np.load('data/body_models/misc/J_regressors.npz'))
        self.smpl_init_sdf = np.load(os.path.join(self.dataset_folder, subjects[0], 'smpl_sdf.npy'), allow_pickle=True).item()
        with open(os.path.join(dataset_folder, subjects[0], 'cam_params.json'), 'r') as f:
            self.cameras = json.load(f)
        if len(views) == 0:
            self.cam_names = self.cameras['all_cam_names']
        else:
            self.cam_names = views
        
        # rotation for da-pose
        self.rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
        self.rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

        # ktree
        self.ktree_parents = np.array([-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,
            9,  9,  9, 12, 13, 14, 16, 17, 18, 19, 20, 21], dtype=np.int32)
        self.ktree_children = np.array([-1,  4,  5,  6,  7,  8,  9,  10,  11,  -1,  -1,  -1,
            15,  16,  17, -1, 18, 19, 20, 21, 22, 23, -1, -1], dtype=np.int32)

        # construct data
        self.data = []
        for subject in subjects:
            subject_dir = os.path.join(dataset_folder, subject)

            if end_frame > 0:
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'models/*.npz')))[start_frame:end_frame:sampling_rate]
            else:
                model_files = sorted(glob.glob(os.path.join(subject_dir, 'models/*.npz')))[start_frame::sampling_rate]

            for cam_idx, cam_name in enumerate(self.cam_names):
                cam_dir = os.path.join(subject_dir, cam_name)

                img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))
                frames = np.arange(len(img_files)).tolist()

                if end_frame > 0:
                    img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[start_frame:end_frame:sampling_rate]
                    mask_files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))[start_frame:end_frame:sampling_rate]
                    frames = frames[start_frame:end_frame:sampling_rate]
                else:
                    img_files = sorted(glob.glob(os.path.join(cam_dir, '*.jpg')))[start_frame::sampling_rate]
                    mask_files = sorted(glob.glob(os.path.join(cam_dir, '*.png')))[start_frame::sampling_rate]
                    frames = frames[start_frame::sampling_rate]

                assert (len(model_files) == len(img_files) and len(mask_files) == len(img_files))

                for d_idx, (f_idx, img_file, mask_file, model_file) in enumerate(zip(frames, img_files, mask_files, model_files)):
                    self.data.append({'subject': subject,
                                    'gender': 'neutral',
                                    'cam_idx': cam_idx,
                                    'cam_name': cam_name,
                                    'frame_idx': f_idx,
                                    'data_idx': d_idx,
                                    'img_file': img_file,
                                    'mask_file': mask_file,
                                    'model_file': model_file})
        
        print(f"Loaded ZJUMOCAP {mode} dataset, the total number of data is {len(self.data)}.")

    def init_grid_homo_2d(self, height, width):
        Y, X = np.meshgrid(np.arange(height, dtype=np.float32),
                        np.arange(width, dtype=np.float32),
                        indexing='ij')
        xy = np.stack([X, Y], axis=-1)  # (height, width, 2)
        H, W = xy.shape[0], xy.shape[1]
        homo_ones = np.ones((H, W, 1), dtype=np.float32)
        homo_2d = np.concatenate((xy, homo_ones), axis=2)
        
        return homo_2d

    def normalize_vectors(self, x):
        norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        eps = 1e-12
        x = x / (norm + eps)
        return x

    def get_camera_rays(self, R, homo_2d):
        rays = np.dot(homo_2d, R) # (H*W, 3)
        rays = self.normalize_vectors(rays) # (H*W, 3)
        return rays

    def get_mask(self, mask_in):
        mask = (mask_in != 0).astype(np.uint8)

        if self.erode_mask or self.mode in ['val', 'test']:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            mask_erode = cv2.erode(mask.copy(), kernel)
            mask_dilate = cv2.dilate(mask.copy(), kernel)
            mask[(mask_dilate - mask_erode) == 1] = 100

        return mask * 255

    def get_posed_shape(self, trans, minimal_shape, bone_transforms, pose_mat, gender):
        posedir = self.posedirs[gender]
        J_regressor = self.J_regressor[gender]
        skinning_weights = self.skinning_weights[gender].astype(np.float32)
        n_smpl_points = minimal_shape.shape[0]
        Jtr = np.dot(J_regressor, minimal_shape) # (24, 3)

        # pose blend shapes
        pose_feature = (pose_mat[1:, ...] - np.eye(3)).reshape([207, 1])
        pose_offsets = np.dot(posedir.reshape([-1, 207]), pose_feature).reshape([6890, 3])
        minimal_shape += pose_offsets

        # posed minimally-clothed shape
        T = np.dot(skinning_weights, bone_transforms.reshape([-1, 16])).reshape([-1, 4, 4])
        homogen_coord = np.ones([n_smpl_points, 1], dtype=np.float32)
        a_pose_homo = np.concatenate([minimal_shape, homogen_coord], axis=-1).reshape([n_smpl_points, 4, 1])
        posed_vertices = (np.matmul(T, a_pose_homo)[:, :3, 0].astype(np.float32) + trans).astype(np.float32)
    
        return Jtr, skinning_weights,  posed_vertices
    
    def load_3d_model(self, data_path):
        model_dict = np.load(data_path)
        trans = model_dict['trans'].astype(np.float32)
        minimal_shape = model_dict['minimal_shape'].astype(np.float32)
        bone_transforms = model_dict['bone_transforms'].astype(np.float32)
        root_orient = model_dict['root_orient'].astype(np.float32)
        pose_body = model_dict['pose_body'].astype(np.float32)
        pose_hand = model_dict['pose_hand'].astype(np.float32)
        Jtr_posed = model_dict['Jtr_posed'].astype(np.float32)

        # body pose
        pose_vec = np.concatenate([root_orient, pose_body, pose_hand], axis=-1) # (72,)
        pose_mat = Rotation.from_rotvec(pose_vec.reshape([-1, 3])).as_matrix() # (24, 3, 3)
        
        return trans, minimal_shape, bone_transforms, pose_mat, pose_vec

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.data)
    
    def __getitem__(self, idx):
        # meta info
        data_path = self.data[idx]['model_file']
        img_path = self.data[idx]['img_file']
        mask_path = self.data[idx]['mask_file']
        cam_name = self.data[idx]['cam_name']
        gender = self.data[idx]['gender']

        # load ext/int matrix and get camera center
        K = np.array(self.cameras[cam_name]['K'], dtype=np.float32)
        D = np.array(self.cameras[cam_name]['D'], dtype=np.float32).ravel()
        R = np.array(self.cameras[cam_name]['R'], np.float32)
        cam_trans = np.array(self.cameras[cam_name]['T'], np.float32).ravel()
        cam_loc = np.dot(-R.T, cam_trans)
        
        # load image/mask and undistortion
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_erode = self.get_mask(mask)
        image = cv2.undistort(image, K, D, None)
        mask = cv2.undistort(mask, K, D, None)
        mask_erode = cv2.undistort(mask_erode, K, D, None)
        img_crop = cv2.resize(image, (self.img_size[1],  self.img_size[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.
        mask_crop = cv2.resize(mask, (self.img_size[1],  self.img_size[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)/255.
        mask_erode_crop = cv2.resize(mask_erode, (self.img_size[1],  self.img_size[0]), interpolation=cv2.INTER_NEAREST).astype(np.float32)/255.
        
        # Update camera parameters for scale
        principal_point = K[:2, -1].reshape(-1).astype(np.float32)
        focal_length = np.array([K[0, 0], K[1, 1]], dtype=np.float32)
        focal_length = focal_length / max(image.shape)  * max(self.img_size)
        principal_point = principal_point / max(image.shape) * max(self.img_size)
        K[:2, -1] = principal_point
        K[0, 0] = focal_length[0]
        K[1, 1] = focal_length[1]
        K_inv = np.linalg.inv(K)    # for mapping rays from camera space to world space

        # 3D models and points
        trans, minimal_shape, bone_transforms, pose_mat, pose_vec = self.load_3d_model(data_path)
        Jtr, skinning_weights, posed_vertices = self.get_posed_shape(trans, minimal_shape, bone_transforms, pose_mat, gender)
        
        # final bone transforms that transforms the canonical Vitruvian-pose mesh to the posed mesh, without global translation
        bone_transforms_02v = get_02v_bone_transforms(Jtr, rot45n=self.rot45n, rot45p=self.rot45p)
        bone_transforms = np.matmul(bone_transforms, np.linalg.inv(bone_transforms_02v))  
        
        # rays sampling
        fg_sample_mask = mask_erode_crop == 1
        bg_sample_mask = mask_erode_crop == 0
        if self.mode == 'train':
            # Get foreground mask bounding box from which to sample rays
            min_xyz = np.min(posed_vertices, axis=0) - self.box_margin
            max_xyz = np.max(posed_vertices, axis=0) + self.box_margin

            bounds = np.stack([min_xyz, max_xyz], axis=0)
            bound_mask = get_bound_2d_mask(bounds, K, np.concatenate([R, cam_trans.reshape([3, 1])], axis=-1), self.img_size[0], self.img_size[1])
            y_inds_bbox, x_inds_bbox = np.where(bound_mask != 0)

            if self.sampling == 'default':
                # Sample foreground pixels
                y_inds, x_inds = np.where(fg_sample_mask)
                fg_inds = np.random.choice(x_inds.shape[0], size=self.num_fg_samples, replace=False)
                y_inds, x_inds = y_inds[fg_inds], x_inds[fg_inds]
                fg_pixels = img_crop[y_inds, x_inds, :].copy()
                fg_mask = mask_crop[y_inds, x_inds].copy()
                fg_mask_erode = mask_erode_crop[y_inds, x_inds].copy()
                fg_uv = np.dot(self.homo_2d.copy()[y_inds, x_inds].reshape([-1, 3]), K_inv.T)

                # Sample background pixels
                inds_mask = bg_sample_mask[y_inds_bbox, x_inds_bbox]
                y_inds = y_inds_bbox[inds_mask]
                x_inds = x_inds_bbox[inds_mask]
                bg_inds = np.random.choice(x_inds.shape[0], size=self.num_bg_samples, replace=False)
                y_inds, x_inds = y_inds[bg_inds], x_inds[bg_inds]
                bg_pixels = np.zeros([x_inds.shape[0], 3], dtype=np.float32)
                bg_mask = mask_crop[y_inds, x_inds].copy()
                bg_mask_erode = mask_erode_crop[y_inds, x_inds].copy()
                bg_uv = np.dot(self.homo_2d.copy()[y_inds, x_inds].reshape([-1, 3]), K_inv.T)

                # compose bg and fg samples
                sampled_pixels = np.concatenate([fg_pixels, bg_pixels], axis=0)
                sampled_mask = np.concatenate([fg_mask, bg_mask], axis=0) != 0
                sampled_mask_erode = np.concatenate([fg_mask_erode, bg_mask_erode], axis=0) != 0
                sampled_uv = np.concatenate([fg_uv, bg_uv], axis=0)
                sampled_rays_cam = self.normalize_vectors(sampled_uv)
                sampled_rays = self.get_camera_rays(R, sampled_uv)
                sampled_near, sampled_far, mask_at_box = get_near_far(bounds, np.broadcast_to(cam_loc, sampled_rays.shape), sampled_rays)
            else:
                raise ValueError('Sampling strategy {} is not supported!'.format(self.sampling))
        else:
            # Test/validation mode
            # Get foreground mask bounding box from which to sample rays
            min_xyz = np.min(posed_vertices, axis=0) - self.box_margin
            max_xyz = np.max(posed_vertices, axis=0) + self.box_margin

            bounds = np.stack([min_xyz, max_xyz], axis=0)
            bound_mask = get_bound_2d_mask(bounds, K, np.concatenate([R, cam_trans.reshape([3, 1])], axis=-1), self.img_size[0], self.img_size[1])
            y_inds, x_inds = np.where(bound_mask != 0)

            sampled_pixels = img_crop[y_inds, x_inds, :].copy()
            sampled_mask = np.ones(sampled_pixels.shape[0], dtype=bool)
            sampled_mask_erode = np.ones(sampled_pixels.shape[0], dtype=bool)
            sampled_bg_mask = bg_sample_mask[y_inds, x_inds].copy()
            sampled_pixels[sampled_bg_mask] = 0
            sampled_uv = np.dot(self.homo_2d.copy()[y_inds, x_inds].reshape([-1, 3]), K_inv.T)
            sampled_rays_cam = self.normalize_vectors(sampled_uv)
            sampled_rays = self.get_camera_rays(R, sampled_uv)

            near, far, mask_at_box = get_near_far(bounds, np.broadcast_to(cam_loc, sampled_rays.shape), sampled_rays)

            sampled_pixels = sampled_pixels[mask_at_box, ...]
            sampled_mask = sampled_mask[mask_at_box, ...]
            sampled_mask_erode = sampled_mask_erode[mask_at_box, ...]
            sampled_uv = sampled_uv[mask_at_box, ...]
            sampled_rays_cam = sampled_rays_cam[mask_at_box, ...]
            sampled_rays = sampled_rays[mask_at_box, ...]
            sampled_near = near[mask_at_box]
            sampled_far = far[mask_at_box]

            image_mask = np.zeros(mask_crop.shape, dtype=bool)
            image_mask[y_inds[mask_at_box], x_inds[mask_at_box]] = True

        # ray mesh intersections computing
        near_bbox_points = sampled_rays * sampled_near[..., None] + cam_loc
        far_bbox_points = sampled_rays * sampled_far[..., None] + cam_loc
        rays_o = np.concatenate([near_bbox_points, far_bbox_points])
        rays_d = np.concatenate([sampled_rays, -sampled_rays])
        hit_mask, _, _, distances = rays_mesh_intersections_pcu(rays_o=rays_o, rays_d=rays_d, vertices=posed_vertices, faces=self.faces)
        hit_mask = hit_mask[:len(sampled_rays)] * hit_mask[len(sampled_rays):]
        # compose hit distances
        distances_near = (distances[:len(sampled_rays)] + sampled_near)[hit_mask]
        distances_far = (distances[len(sampled_rays):] * -1 + sampled_far)[hit_mask]
        sampled_near[hit_mask] = distances_near
        sampled_far[hit_mask] = distances_far
        
        if DEBUG:
            np.savez(
                "rays",
                image=image,
                mask_erode=mask_erode,
                idx=idx,
                cam_loc=cam_loc,
                fg_inds=fg_inds,
                bg_inds=bg_inds,
                sampled_rays=sampled_rays,
            )
            trimesh.Trimesh(vertices=posed_vertices, faces=self.faces).export("posed_vertices.obj")
        
        data = {
            "color": sampled_pixels,
            "mask": sampled_mask, 
            "mask_erode": sampled_mask_erode,
            "rays_d": sampled_rays,
            "rays_o": cam_loc,
            "near": sampled_near,
            "far": sampled_far,
            "skinning_weights": skinning_weights,
            "posed_vertices":  posed_vertices,
            "faces": self.faces,
            "bone_transforms": bone_transforms.astype(np.float32),
            
            "trans": trans,
            "sdf_init_kwargs": self.smpl_init_sdf,
            "dst_posevec": pose_vec[3:] + 1e-2, # ndarray (69,)
        }
        if self.mode !='train':
            data.update({
                'image_mask': image_mask,
                'image_size': self.img_size,
                'R': R
            })
        
        return data
