import os
import pickle
import numpy as np
import cv2
import logging
from tqdm import tqdm
import torch
import torch.utils.data
import trimesh
from scipy.spatial.transform import Rotation

from libs.utils.general_utils import rays_mesh_intersections_pcu, get_02v_bone_transforms, get_z_vals
from libs.utils.images_utils import load_image
from libs.utils.body_utils import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from libs.utils.file_utils import list_files, split_path
from libs.utils.camera_utils import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

class PeopleSnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset_folder='data/data_prepared',
                 subjects=['CoreView_313'],
                 mode='train',
                 resize_img_scale=1.,
                 start_frame=0,
                 end_frame=-1,
                 sampling_rate=1,
                 views=[0, 6, 12, 18],
                 box_margin=0.05,
                 ray_shoot_mode='default',
                 backgroung_color=None,
                 patch_size=32,
                 N_patches=1,
                 sample_subject_ratio=0.8,
                 N_samples=64,
                 inner_sampling=False,
                 res_level=4,
                 use_dilated=False,
                 use_inter_mask=False):
        assert len(subjects) == 1, 'SINGLE PERSON! Please make sure the length of subjects list is one.'
        
        # meta data
        self.mode = mode
        self.views = views
        self.use_dilated = use_dilated
        self.patch_size = patch_size
        self.N_patches = N_patches
        self.sample_subject_ratio = sample_subject_ratio
        self.N_samples = N_samples
        self.volume_size = 64
        self.inner_sampling = inner_sampling
        self.use_inter_mask = use_inter_mask
        self.res_level = res_level
        self.bgcolor = np.array(backgroung_color, dtype=np.float32) if backgroung_color is not None else (np.random.rand(3) * 255).astype(np.float32)
        self.ray_shoot_mode = ray_shoot_mode
        self.bbox_offset=box_margin
        self.resize_img_scale=resize_img_scale
        self.dataset_path = os.path.join(dataset_folder, subjects[0])
        
        # load data
        self.canonical_joints, self.canonical_bbox = self.load_canonical_joints()
        self.gtfs_02v = get_02v_bone_transforms(self.canonical_joints)
        self.cnl_gtfms = get_canonical_global_tfms(self.canonical_joints) # (24, 4, 4)
        self.mesh_infos = self.load_train_mesh_infos()
        self.smpl_sdf = self.load_smpl_sdf()
        self.cameras_dict = self.load_train_cameras()
        self.framelist_dict, self.nframes_dict, self.nframes_total = \
                self.load_train_frames(start_frame, end_frame, sampling_rate)
        self.faces = np.load('data/body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('data/body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('data/body_models/misc/posedirs_all.npz'))
        self.J_regressor = dict(np.load('data/body_models/misc/J_regressors.npz'))

        images, masks = {}, {}
        for view in self.views:
            framelist = self.framelist_dict[view]
            images_list, masks_list = [], []
            for fname in tqdm(framelist):
                image, mask = self.load_image_mask(view, fname, self.bgcolor) # (H, W, 3) (H, W, 1)
                images_list.append(image/255)
                masks_list.append(mask)
            images[view] = np.stack(images_list).astype(np.float32) # (n_images, H, W, 3)
            masks[view] = np.stack(masks_list).astype(np.float32) # (n_images, H, W, 1)
        self.images_np_dict = images
        self.masks_np_dict = masks
        
        logging.info(f'[Dataset Path]: {self.dataset_path} / {self.mode}. -- Total Frames: {self.__len__()}')
        
    def load_smpl_sdf(self):
        return np.load(os.path.join(self.dataset_path, 'smpl_sdf.npy'), allow_pickle=True).item()
    
    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        # canonical_bbox = self.skeleton_to_bbox(canonical_joints)
        canonical_vertices = cl_joint_data['vertices'].astype('float32')
        canonical_bbox = self.vertices_to_bbox(canonical_vertices)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        cameras_dict = {}
        for view in self.views:
            camera_info_dir = os.path.join(self.dataset_path, view)
            with open(os.path.join(camera_info_dir, 'cameras.pkl'), 'rb') as f: 
                cameras_data = pickle.load(f)
            cameras_dict[view] = cameras_data
        return cameras_dict

    def skeleton_to_bbox(self, skeleton):
        min_xyz = np.min(skeleton, axis=0) - self.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + self.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }
    
    def vertices_to_bbox(self, vertices):
        min_xyz = np.min(vertices, axis=0) - self.bbox_offset
        max_xyz = np.max(vertices, axis=0) + self.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            # bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['posed_vertices'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos

    def load_train_frames(self, start_frame, end_frame, sampling_rate):
        img_path_dict = {}
        nframe_dict = {}
        nframe_total = 0
        for view in self.views:
            image_dir = os.path.join(self.dataset_path, view, 'images')
            img_paths = list_files(image_dir, exts=['.png'])
            img_path_dict[view] = [split_path(ipath)[1] for ipath in img_paths][start_frame:end_frame:sampling_rate]
            nframe_dict[view] = len(img_path_dict[view])
            nframe_total += len(img_path_dict[view])
        return img_path_dict, nframe_dict, nframe_total
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            # 'dst_tpose_joints': \
            #     self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'dst_tpose_joints': self.canonical_joints.astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32'),
            'posed_vertices': self.mesh_infos[frame_name]['posed_vertices'].astype('float32'),
            'dilated_vertices': self.mesh_infos[frame_name]['dilated_vertices'].astype('float32'),
            'dilated_triangles': self.mesh_infos[frame_name]['dilated_triangles'].copy(),
        }
    
    def load_image_mask(self, view, frame_name, bg_color):
        imagepath = os.path.join(self.dataset_path, view, 'images', '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path, view, 'masks','{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))[...,:1]
        
        # undistort image
        if frame_name in self.cameras_dict[view] and 'distortions' in self.cameras_dict[view][frame_name]:
            K = self.cameras_dict[view][frame_name]['intrinsics']
            D = self.cameras_dict[view][frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)[..., None]
        
        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        # a = img.astype(np.uint8)
        if self.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=self.resize_img_scale,
                                fy=self.resize_img_scale,
                                interpolation=cv2.INTER_LINEAR)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=self.resize_img_scale,
                                    fy=self.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)[..., None]
        
        return img, alpha_mask

    def sample_batch_rays(self, N_rays, rays_o, rays_d, rays_img, rays_alpha, near, far):
        sample_idxs=np.random.randint(0, rays_o.shape[0], N_rays)
        rays_o = rays_o[sample_idxs].astype(np.float32) # (N_rays, 3)
        rays_d = rays_d[sample_idxs].astype(np.float32) # (N_rays, 3)
        rays_img = rays_img[sample_idxs].astype(np.float32) # (N_rays, 3)
        rays_alpha = rays_alpha[sample_idxs].astype(np.float32) # (N_rays, 1)
        near = near[sample_idxs].astype(np.float32) # (N_rays, 1)
        far = far[sample_idxs].astype(np.float32) # (N_rays, 1)
        
        return rays_o, rays_d, rays_img, rays_alpha, near, far
    
    def sample_image_rays(self, res_level, rays_o, rays_d, rays_img, rays_alpha, near, far):
        rays_o = rays_o[::res_level, ::res_level].reshape(-1, 3) # (N_rays, 3)
        rays_d = rays_d[::res_level, ::res_level].reshape(-1, 3) # (N_rays, 3)
        rays_img = rays_img[::res_level, ::res_level].reshape(-1, 3) # (N_rays, 3)
        rays_alpha = rays_alpha[::res_level, ::res_level].reshape(-1, 1) # (N_rays, 1)
        near = near[::res_level, ::res_level].reshape(-1, 1) # (N_rays, 1)
        far = far[::res_level, ::res_level].reshape(-1, 1) # (N_rays, 1)
        
        return rays_o, rays_d, rays_img, rays_alpha, near, far

    def random_selected_view_idx(self, idx):
        sel_idx = idx
        for view in self.views:
            if sel_idx - self.nframes_dict[view] < 0:
                sel_view = view
                break
            else:
                sel_idx = sel_idx - self.nframes_dict[view]
        return sel_view, sel_idx
    
    def gen_rays_for_infer(self, idx):
        view, idx = self.random_selected_view_idx(idx)
        res_level = self.res_level
        
        # get image & mask
        frame_name = self.framelist_dict[view][idx]
        image = self.images_np_dict[view][idx] # (H, W, 3)
        alpha = self.masks_np_dict[view][idx] # (H, W, 1)
        H, W = image.shape[0:2]

        # dst skeleton info
        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox'] 
        dst_poses = dst_skel_info['poses'] # (24 x 3,)
        dst_tpose_joints = dst_skel_info['dst_tpose_joints'] # (24, 3)
        dst_vertices = dst_skel_info['posed_vertices']
        dst_Rs, dst_Ts = body_pose_to_body_RTs(dst_poses, dst_tpose_joints) # (24, 3, 3) (24, 3)

        # calculate K, E, T
        assert frame_name in self.cameras_dict[view]
        K = self.cameras_dict[view][frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= self.resize_img_scale
        E = self.cameras_dict[view][frame_name]['extrinsics'].copy()
        E = apply_global_tfm_to_camera(
                E=E,
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th']+np.array([0,-0.4,0])) # TODO: [0,-0.5,0] gained by experience
        R = E[:3, :3]
        T = E[:3, 3:]

        # calculate rays in world coordinate from pixel/image coordinate
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)

        # calculate rays and near & far which intersect with cononical space bbox
        near_, far_, rays_mask = rays_intersect_3d_bbox(dst_bbox, rays_o.reshape(-1, 3), rays_d.reshape(-1, 3))
        # # for debug
        # aa = rays_mask.reshape(1080, 1080)
        # cv2.imwrite("aa.jpg", aa.astype(np.uint8)*255)
        # cv2.imwrite("bb.jpg", image*255)
        
        near = np.ones(H*W) * near_.mean()
        near[rays_mask] = near_
        far = np.ones(H*W) * far_.mean()
        far[rays_mask] = far_
        
        # calculate batch or patch rays
        rays_o, rays_d, rays_img, rays_alpha, near, far = self.sample_image_rays(res_level=res_level,
                            rays_o=rays_o,
                            rays_d=rays_d,
                            rays_img=image,
                            rays_alpha=alpha,
                            near=near.reshape(H, W, 1),
                            far=far.reshape(H, W, 1)) # (N_rays, 12) denote there are N_rays rays
        batch_rays = np.concatenate([rays_o, rays_d, rays_img, rays_alpha, near, far], axis=-1) # (N_rays, 12)
        
        # return result
        min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
        max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
        results={
            'view': view,
            'idx': idx,
            'batch_rays': batch_rays.astype(np.float32), # ndarray (N_rays, 12)
            'bbox_mask': rays_mask.reshape(H, W)[::res_level, ::res_level], # ndarray (H, W, 1)
            
            'frame_name': frame_name, # str
            'extrinsic': E.astype(np.float32), # ndarray (4, 4)
            'intrinsic': K.astype(np.float32), # ndarray (3, 3)
            'width': W//res_level, # int
            'height': H//res_level, # int
            'background_color': self.bgcolor, # ndarray (3, )
            'tjoints': self.canonical_joints, # ndarray (24, 3)
            'gtfs_02v': self.gtfs_02v.astype(np.float32), # ndarray (24, 4, 4)
            'dst_Rs': dst_Rs, # ndarray (24, 3, 3)
            'dst_Ts': dst_Ts, # ndarray (24, 3)
            'dst_posevec': dst_poses[3:] + 1e-2, # ndarray (69,)
            'dst_vertices': dst_vertices, # ndarray (6890, 3),
            'cnl_gtfms': self.cnl_gtfms.astype(np.float32), # ndarray (24, 4, 4)
            'cnl_bbox_min_xyz': min_xyz, # ndarray (3,)
            'cnl_bbox_max_xyz': max_xyz, # ndarray (3,)
            'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz),  # ndarray (3,)
            'skinning_weights': self.skinning_weights['neutral'].astype(np.float32), # ndarray (6890, 24)
            'smpl_sdf' :{
                'bbmin': self.smpl_sdf['bbmin'].astype(np.float32),
                'bbmax': self.smpl_sdf['bbmax'].astype(np.float32),
                'sdf_grid': self.smpl_sdf['sdf_grid'].astype(np.float32),
            },
            'geo_latent_code_idx': idx,
            'camera_e':self.cameras_dict[view][frame_name]['extrinsics'].astype(np.float32),
            'rh': cv2.Rodrigues(dst_skel_info['Rh'].copy().astype(np.float32))[0].T,
            'th': dst_skel_info['Th'].copy().astype(np.float32)+np.array([0,-0.4,0]),
            # 'rots': pose_rot.astype(np.float32), # (24, 9), pose rotation, where the root rotation is identity
            # 'Jtrs': Jtr_norm.astype(np.float32), # (24 3), T-pose joint points
        }
        if self.inner_sampling:
            if self.use_dilated:
                dilated_vertices = dst_skel_info['dilated_vertices']
                dilated_triangles = dst_skel_info['dilated_triangles']
                z_vals, inter_mask = get_z_vals(near, far, rays_o, rays_d, dilated_vertices, dilated_triangles, self.N_samples)
            else:
                z_vals, inter_mask = get_z_vals(near, far, rays_o, rays_d, dst_vertices, self.faces, self.N_samples)
            
            results['z_vals'] = z_vals.astype(np.float32)
            results['hit_mask'] = np.logical_or(inter_mask, rays_alpha.reshape(-1).astype(np.bool_))
        else:
            t_vals = np.linspace(0.0, 1.0, self.N_samples)
            z_vals = near + (far - near) * t_vals[None, :]
            results['z_vals'] = z_vals.astype(np.float32)
            results['hit_mask'] = rays_alpha.reshape(-1).astype(np.bool_)
            
        return results
    
    def gen_rays_for_train(self, idx):
        view, idx = self.random_selected_view_idx(idx)
        
        # get image & mask
        frame_name = self.framelist_dict[view][idx]
        image = self.images_np_dict[view][idx] # (H, W, 3)
        alpha = self.masks_np_dict[view][idx] # (H, W, 1)
        H, W = image.shape[0:2]

        # dst skeleton info
        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox']
        dst_poses = dst_skel_info['poses'] # (24 x 3,)
        dst_tpose_joints = dst_skel_info['dst_tpose_joints'] # (24, 3)
        dst_vertices = dst_skel_info['posed_vertices']
        
        dst_Rs, dst_Ts = body_pose_to_body_RTs(dst_poses, dst_tpose_joints) # (24, 3, 3) (24, 3)
        
        # calculate K, E, T
        assert frame_name in self.cameras_dict[view]
        K = self.cameras_dict[view][frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= self.resize_img_scale

        E = self.cameras_dict[view][frame_name]['extrinsics'].copy()
        E = apply_global_tfm_to_camera(
                E=E,
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th']+np.array([0,-0.4,0])) # TODO: [0,-0.5,0] gained by experience
        R = E[:3, :3]
        T = E[:3, 3:]

        # calculateinfo rays in world coordinate from pixel/image coordinate
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        rays_img = image.reshape(-1, 3) # (H, W, 3) --> (HxW, 3)
        rays_alpha = alpha.reshape(-1, 1) # (H, W, 1) --> (HxW, 1)
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (HxW, 3)
        rays_d = rays_d.reshape(-1, 3) # (H, W, 3) --> (HxW, 3)

        # calculate rays and near & far which intersect with cononical space bbox
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        # # for debug
        # aa = ray_mask.reshape(1080, 1080)
        # cv2.imwrite("aa.jpg", aa.astype(np.uint8)*255)
        # cv2.imwrite("bb.jpg", image*255)
        
        rays_o = rays_o[ray_mask] # (len(ray_mask), 3)
        rays_d = rays_d[ray_mask] # (len(ray_mask), 3)
        rays_img = rays_img[ray_mask] # (len(ray_mask), 3)
        rays_alpha = rays_alpha[ray_mask] # (len(ray_mask), 1)
        near = near[:, None].astype('float32') # (len(ray_mask), 1)
        far = far[:, None].astype('float32') # (len(ray_mask), 1)

        if self.ray_shoot_mode == 'default':
            # calculate batch or patch rays
            rays_o, rays_d, rays_img, rays_alpha, near, far = \
                self.sample_batch_rays(N_rays=self.N_patches * self.patch_size**2,
                                rays_o=rays_o,
                                rays_d=rays_d,
                                rays_img=rays_img,
                                rays_alpha=rays_alpha,
                                near=near,
                                far=far)
        elif self.ray_shoot_mode == 'patch':
            rays_o, rays_d, rays_img, rays_alpha, near, far, target_patches, patch_masks, patch_div_indices = \
            self.sample_patch_rays(image=image, H=H, W=W,
                                subject_mask=alpha[:, :, 0] > 0.,
                                bbox_mask=ray_mask.reshape(H, W),
                                ray_mask=ray_mask,
                                rays_o=rays_o, 
                                rays_d=rays_d, 
                                rays_img=rays_img, 
                                rays_alpha=rays_alpha,
                                near=near,
                                far=far)
        else:
            raise ValueError('Sampling strategy {} is not supported!'.format(self.sampling))
        
        batch_rays = np.concatenate([rays_o, rays_d, rays_img, rays_alpha, near, far], axis=-1) # (N_rays, 12)
        
        if False:
            import trimesh
            points = z_vals[..., None] * rays_d[:, None] + rays_o[:, None]
            trimesh.Trimesh(points[inter_mask].reshape(-1, 3)).export("points.obj")
        
        min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
        max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
        results={
            'batch_rays': batch_rays.astype(np.float32), # ndarray (N_rays, 12)
            
            'frame_name': frame_name, # str
            'width': W, # int
            'height': H, # int
            'extrinsic': E.astype(np.float32), # ndarray (4, 4)
            'intrinsic': K.astype(np.float32), # ndarray (3, 3)
            'background_color': self.bgcolor, # ndarray (3, )
            'tjoints': self.canonical_joints,
            'gtfs_02v': self.gtfs_02v.astype(np.float32), # ndarray (24, 4, 4)
            'dst_Rs': dst_Rs, # ndarray (24, 3, 3)
            'dst_Ts': dst_Ts, # ndarray (24, 3)
            'dst_posevec': dst_poses[3:] + 1e-2, # ndarray (69,)
            'dst_vertices': dst_vertices, # ndarray (6890, 3)
            'cnl_gtfms': self.cnl_gtfms.astype(np.float32), # ndarray (24, 4, 4)
            'cnl_bbox_min_xyz': min_xyz, # ndarray (3,)
            'cnl_bbox_max_xyz': max_xyz, # ndarray (3,)
            'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz),  # ndarray (3,)
            'skinning_weights': self.skinning_weights['neutral'].astype(np.float32), # ndarray (6890, 24)
            'smpl_sdf' :{
                'bbmin': self.smpl_sdf['bbmin'].astype(np.float32),
                'bbmax': self.smpl_sdf['bbmax'].astype(np.float32),
                'sdf_grid': self.smpl_sdf['sdf_grid'].astype(np.float32),
            },
            'geo_latent_code_idx': idx,
            'camera_e':self.cameras_dict[view][frame_name]['extrinsics'].astype(np.float32),
            'rh': cv2.Rodrigues(dst_skel_info['Rh'].copy().astype(np.float32))[0].T,
            'th': dst_skel_info['Th'].copy().astype(np.float32),
        }
        if self.ray_shoot_mode == 'patch':
            results.update({
                'target_patches': target_patches,
                'patch_masks': patch_masks,
                'patch_div_indices': patch_div_indices,
                })
        if self.inner_sampling:
            if self.use_dilated:
                dilated_vertices = dst_skel_info['dilated_vertices']
                dilated_triangles = dst_skel_info['dilated_triangles']
                z_vals, inter_mask = get_z_vals(near, far, rays_o, rays_d, dilated_vertices, dilated_triangles, self.N_samples)
            else:
                z_vals, inter_mask = get_z_vals(near, far, rays_o, rays_d, dst_vertices, self.faces, self.N_samples)
            
            results['z_vals'] = z_vals.astype(np.float32)
            results['hit_mask'] = np.logical_or(inter_mask, rays_alpha.reshape(-1).astype(np.bool_))
        else:
            t_vals = np.linspace(0.0, 1.0, self.N_samples)
            z_vals = near + (far - near) * t_vals[None, :]
            results['z_vals'] = z_vals.astype(np.float32)
            results['hit_mask'] = rays_alpha.reshape(-1).astype(np.bool_)
            
        return results
    
    def __len__(self):
        return self.nframes_total
    
    def __getitem__(self, idx):
        if self.mode in ['test','val']:
            return self.gen_rays_for_infer(idx)
        else:
            return self.gen_rays_for_train(idx)
    
    def update_near_far(self, sampled_near, sampled_far, cam_loc, sampled_rays, posed_vertices):
        # ray mesh intersections computing
        near_bbox_points = sampled_rays * sampled_near + cam_loc
        far_bbox_points = sampled_rays * sampled_far + cam_loc
        rays_o = np.concatenate([near_bbox_points, far_bbox_points])
        rays_d = np.concatenate([sampled_rays, -sampled_rays])
        hit_mask, _, _, distances = rays_mesh_intersections_pcu(rays_o=rays_o, rays_d=rays_d, vertices=posed_vertices, faces=self.faces)
        hit_mask = hit_mask[:len(sampled_rays)] * hit_mask[len(sampled_rays):]
        # compose hit distances
        distances_near = (distances[:len(sampled_rays)][..., None] + sampled_near)[hit_mask]
        distances_far = (distances[len(sampled_rays):][..., None] * -1 + sampled_far)[hit_mask]
        sampled_near[hit_mask] = distances_near
        sampled_far[hit_mask] = distances_far
        
        return sampled_near, sampled_far, hit_mask
        
    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool_
        assert candidate_mask.dtype == np.bool_

        valid_ys, valid_xs = np.where(candidate_mask)

        # determine patch center
        select_idx = np.random.choice(valid_ys.shape[0], 
                                      size=[1], replace=False)[0]
        center_x = valid_xs[select_idx]
        center_y = valid_ys[select_idx]

        # determine patch boundary
        half_patch_size = patch_size // 2
        x_min = np.clip(a=center_x-half_patch_size, 
                        a_min=0, 
                        a_max=W-patch_size)
        x_max = x_min + patch_size
        y_min = np.clip(a=center_y-half_patch_size,
                        a_min=0,
                        a_max=H-patch_size)
        y_max = y_min + patch_size

        sel_ray_mask = np.zeros_like(candidate_mask)
        sel_ray_mask[y_min:y_max, x_min:x_max] = True

        #####################################################
        ## Below we determine the selected ray indices
        ## and patch valid mask

        sel_ray_mask = sel_ray_mask.reshape(-1)
        inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
        select_masked_inds = np.where(inter_mask)

        masked_indices = np.cumsum(ray_mask) - 1
        select_inds = masked_indices[select_masked_inds]
        
        inter_mask = inter_mask.reshape(H, W)

        return select_inds, \
                inter_mask[y_min:y_max, x_min:x_max], \
                np.array([x_min, y_min]), np.array([x_max, y_max])
    
    def get_patch_ray_indices(
            self, 
            N_patch, 
            ray_mask, 
            subject_mask, 
            bbox_mask,
            patch_size, 
            H, W):

        assert subject_mask.dtype == np.bool_
        assert bbox_mask.dtype == np.bool_

        bbox_exclude_subject_mask = np.bitwise_and(
            bbox_mask,
            np.bitwise_not(subject_mask)
        )

        list_ray_indices = []
        list_mask = []
        list_xy_min = []
        list_xy_max = []

        total_rays = 0
        patch_div_indices = [total_rays]
        for _ in range(N_patch):
            # let p = cfg.patch.sample_subject_ratio
            # prob p: we sample on subject area
            # prob (1-p): we sample on non-subject area but still in bbox
            if np.random.rand(1)[0] < self.sample_subject_ratio:
                candidate_mask = subject_mask
            else:
                candidate_mask = bbox_exclude_subject_mask

            ray_indices, mask, xy_min, xy_max = \
                self._get_patch_ray_indices(ray_mask, candidate_mask, 
                                            patch_size, H, W)

            assert len(ray_indices.shape) == 1
            total_rays += len(ray_indices)

            list_ray_indices.append(ray_indices)
            list_mask.append(mask)
            list_xy_min.append(xy_min)
            list_xy_max.append(xy_max)
            
            patch_div_indices.append(total_rays)

        select_inds = np.concatenate(list_ray_indices, axis=0)
        patch_info = {
            'mask': np.stack(list_mask, axis=0),
            'xy_min': np.stack(list_xy_min, axis=0),
            'xy_max': np.stack(list_xy_max, axis=0)
        }
        patch_div_indices = np.array(patch_div_indices)

        return select_inds, patch_info, patch_div_indices

    def select_rays(self, select_inds, rays_o, rays_d, rays_img, rays_alpha, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        rays_img = rays_img[select_inds]
        rays_alpha = rays_alpha[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, rays_img, rays_alpha, near, far
    
    def sample_patch_rays(self, image, H, W,
                          subject_mask, bbox_mask, ray_mask,
                          rays_o, rays_d, rays_img, rays_alpha, near, far):
        
        select_inds, patch_info, patch_div_indices = \
            self.get_patch_ray_indices(
                N_patch=self.N_patches,
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=self.patch_size, 
                H=H, W=W)

        # rays_o : (N_patches x patch_size x patch_size, 3)
        # rays_d : (N_patches x patch_size x patch_size, 3)
        # rays_img : (N_patches x patch_size x patch_size, 3)
        # rays_alpha : (N_patches x patch_size x patch_size, 1)
        # near : (N_patches x patch_size x patch_size, 1)
        # far : (N_patches x patch_size x patch_size, 1)
        rays_o, rays_d, rays_img, rays_alpha, near, far = self.select_rays(
            select_inds, rays_o, rays_d, rays_img, rays_alpha, near, far) 
        
        targets = []
        for i in range(self.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(image[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, rays_img, rays_alpha, near, far, target_patches, patch_masks, patch_div_indices