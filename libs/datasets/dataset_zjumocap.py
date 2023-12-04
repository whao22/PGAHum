import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.utils.data

from models.utils.images_utils import load_image
from models.utils.body_utils import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from models.utils.file_utils import list_files, split_path
from models.utils.camera_utils import \
    apply_global_tfm_to_camera, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox


class ZJUMoCapDataset(torch.utils.data.Dataset):
    def __init__(self, conf):
        
        # meta info
        self.device = torch.device('cuda')
        self.conf = conf
        self.dataset_path = conf.get_string('data_dir')
        print('[Dataset Path]', self.dataset_path) 
        self.image_dir = os.path.join(self.dataset_path, 'images')
        
        # flags
        self.bgcolor = np.array(conf.backgroung_color, dtype=np.float32) if conf.backgroung_color is not None else (np.random.rand(3) * 255).astype(np.float32)
        self.ray_shoot_mode = conf.ray_shoot_mode
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)
        self.volume_size=conf.volume_size
        self.bbox_offset=conf.bbox_offset # following HFS, we assume the human standing in a unit sphere   
        self.resize_img_scale=conf.resize_img_scale
        conf.batch_size=conf.N_patches * conf.patch_size**2
        
        
        # load data
        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()
        self.motion_weights_priors = \
            approx_gaussian_bone_volumes(
                self.canonical_joints,   
                self.canonical_bbox['min_xyz'],
                self.canonical_bbox['max_xyz'],
                grid_size=self.volume_size).astype('float32')
        self.cameras = self.load_train_cameras()
        self.mesh_infos = self.load_train_mesh_infos()
        self.framelist = self.load_train_frames()[::conf.skip]
        if conf.maxframes > 0:
            self.framelist = self.framelist[:conf.maxframes]
        self.n_images = len(self.framelist)
        
        images, masks = [], []
        for fname in tqdm(self.framelist):
            image, mask = self.load_image(fname, self.bgcolor) # (H, W, 3) (H, W, 1)
            images.append(image/255)
            masks.append(mask)
        self.images_np = np.stack(images).astype(np.float32) # (n_images, H, W, 3)
        self.masks_np = np.stack(masks).astype(np.float32) # (n_images, H, W, 1)
        # log print
        print(f' -- Total Frames: {self.get_total_frames()}')

        
    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox


    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras


    def skeleton_to_bbox(self, skeleton):
        min_xyz = np.min(skeleton, axis=0) - self.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + self.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }


    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos


    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }
    
    
    def load_image(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))
        alpha_mask = np.array(load_image(maskpath))[...,:1]
        
        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
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


    def get_total_frames(self):
        return len(self.framelist)


    def sample_batch_rays(self, batch_size, rays_o, rays_d, rays_img, rays_alpha, near, far):
        N_rays = rays_o.shape[0]
        
        sample_idxs=np.random.randint(0, N_rays, batch_size)
        rays_o = rays_o[sample_idxs] # (batch_size, 3)
        rays_d = rays_d[sample_idxs] # (batch_size, 3)
        rays_img = rays_img[sample_idxs] # (batch_size, 3)
        rays_alpha = rays_alpha[sample_idxs] # (batch_size, 1)
        near = near[sample_idxs] # (batch_size, 1)
        far = far[sample_idxs] # (batch_size, 1)
        return np.concatenate([rays_o, rays_d, rays_img, rays_alpha, near, far], axis=-1) # (batch_size, 12)
    
    
    def sample_image_rays(self, res_level, rays_o, rays_d, rays_img, rays_alpha, near, far):
        rays_o = rays_o[::res_level, ::res_level].reshape(-1, 3) # (batch_size, 3)
        rays_d = rays_d[::res_level, ::res_level].reshape(-1, 3) # (batch_size, 3)
        rays_img = rays_img[::res_level, ::res_level].reshape(-1, 3) # (batch_size, 3)
        rays_alpha = rays_alpha[::res_level, ::res_level].reshape(-1, 1) # (batch_size, 1)
        near = near[::res_level, ::res_level].reshape(-1, 1) # (batch_size, 1)
        far = far[::res_level, ::res_level].reshape(-1, 1) # (batch_size, 1)
        
        return  np.concatenate([rays_o, rays_d, rays_img, rays_alpha, near, far], axis=-1) # (batch_size, 12)


    def gen_rays_at(self, idx, res_level):
        # get image & mask
        frame_name = self.framelist[idx]
        image = self.images_np[idx] # (H, W, 3)
        alpha = self.masks_np[idx] # (H, W, 1)
        H, W = image.shape[0:2]

        # dst skeleton info
        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox'] 
        dst_poses = dst_skel_info['poses'] # (24 x 3,)
        dst_tpose_joints = dst_skel_info['dst_tpose_joints'] # (24, 3)
        dst_Rs, dst_Ts = body_pose_to_body_RTs(dst_poses, dst_tpose_joints) # (24, 3, 3) (24, 3)
        cnl_gtfms = get_canonical_global_tfms(self.canonical_joints) # (24, 4, 4)

        # calculate K, E, T
        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= self.resize_img_scale
        E = self.cameras[frame_name]['extrinsics']
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]

        # calculate rays in world coordinate from pixel/image coordinate 
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T) # (H, W, 3)

        # calculate rays and near & far which intersect with cononical space bbox
        near_, far_, rays_mask = rays_intersect_3d_bbox(dst_bbox, rays_o.reshape(-1, 3), rays_d.reshape(-1, 3))
        near = np.ones(H*W) * near_.mean()
        near[rays_mask] = near_
        far = np.ones(H*W) * far_.mean()
        far[rays_mask] = far_
        
        # calculate batch or patch rays
        batch_rays=self.sample_image_rays(res_level=res_level,
                            rays_o=rays_o,
                            rays_d=rays_d,
                            rays_img=image,
                            rays_alpha=alpha,
                            near=near.reshape(H, W, 1),
                            far=far.reshape(H, W, 1)) # (batch_size, 12) denote there are batch_size rays
        
        # return result
        min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
        max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
        results={
            'batch_rays': batch_rays, # ndarray (batch_size, 12)
            'frame_name': frame_name, # str
            'extrinsic': E, # ndarray (4, 4)
            'width': W//res_level, # int
            'height': H//res_level, # int
            'background_color': self.bgcolor, # ndarray (3, )
            'dst_Rs': dst_Rs, # ndarray (24, 3, 3)
            'dst_Ts': dst_Ts, # ndarray (24, 3)
            'dst_posevec': dst_poses[3:] + 1e-2, # ndarray (69,)
            'cnl_gtfms': cnl_gtfms, # ndarray (24, 4, 4)
            'mweights_priors': self.motion_weights_priors, # ndarray (24, 32, 32, 32)
            'cnl_bbox_min_xyz': min_xyz, # ndarray (3,)
            'cnl_bbox_max_xyz': max_xyz, # ndarray (3,)
            'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)  # ndarray (3,)
        }
        return results


    def gen_random_rays_at(self, idx, batch_size):
        # get image & mask
        frame_name = self.framelist[idx]
        image = self.images_np[idx] # (H, W, 3)
        alpha = self.masks_np[idx] # (H, W, 1)
        H, W = image.shape[0:2]

        # dst skeleton info
        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox'] 
        dst_poses = dst_skel_info['poses'] # (24 x 3,)
        dst_tpose_joints = dst_skel_info['dst_tpose_joints'] # (24, 3)
        
        dst_Rs, dst_Ts = body_pose_to_body_RTs(dst_poses, dst_tpose_joints) # (24, 3, 3) (24, 3)
        cnl_gtfms = get_canonical_global_tfms(self.canonical_joints) # (24, 4, 4)

        # calculate K, E, T
        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= self.resize_img_scale

        E = self.cameras[frame_name]['extrinsics'].copy()
        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])
        R = E[:3, :3]
        T = E[:3, 3]

        # calculate rays in world coordinate from pixel/image coordinate
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        rays_img = image.reshape(-1, 3) # (H, W, 3) --> (HxW, 3)
        rays_alpha = alpha.reshape(-1, 1) # (H, W, 1) --> (HxW, 1)
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (HxW, 3)
        rays_d = rays_d.reshape(-1, 3) # (H, W, 3) --> (HxW, 3)

        # calculate rays and near & far which intersect with cononical space bbox
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask] # (len(ray_mask), 3)
        rays_d = rays_d[ray_mask] # (len(ray_mask), 3)
        rays_img = rays_img[ray_mask] # (len(ray_mask), 3)
        rays_alpha = rays_alpha[ray_mask] # (len(ray_mask), 1)
        near = near[:, None].astype('float32') # (len(ray_mask), 1)
        far = far[:, None].astype('float32') # (len(ray_mask), 1)

        # # calculate batch or patch rays
        # batch_rays=self.sample_batch_rays(batch_size=batch_size,
        #                     rays_o=rays_o,
        #                     rays_d=rays_d,
        #                     rays_img=rays_img,
        #                     rays_alpha=rays_alpha,
        #                     near=near,
        #                     far=far) # (batch_size, 12) denote there are batch_size rays

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
        
        batch_rays = np.concatenate([rays_o, rays_d, rays_img, rays_alpha, near, far], axis=-1)
        
        # return result
        min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
        max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
        results={
            'batch_rays': batch_rays, # ndarray (batch_size, 12)
            'target_patches': target_patches,
            'patch_masks': patch_masks,
            'patch_div_indices': patch_div_indices,
            'frame_name': frame_name, # str
            'width': W, # int
            'height': H, # int
            'background_color': self.bgcolor, # ndarray (3, )
            'dst_Rs': dst_Rs, # ndarray (24, 3, 3)
            'dst_Ts': dst_Ts, # ndarray (24, 3)
            'dst_posevec': dst_poses[3:] + 1e-2, # ndarray (69,)
            'cnl_gtfms': cnl_gtfms, # ndarray (24, 4, 4)
            'mweights_priors': self.motion_weights_priors.copy(), # ndarray (24, 32, 32, 32)
            'cnl_bbox_min_xyz': min_xyz, # ndarray (3,)
            'cnl_bbox_max_xyz': max_xyz, # ndarray (3,)
            'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)  # ndarray (3,)
        }
        return results
    
    
    def _get_patch_ray_indices(
            self, 
            ray_mask, 
            candidate_mask, 
            patch_size, 
            H, W):

        assert len(ray_mask.shape) == 1
        assert ray_mask.dtype == np.bool
        assert candidate_mask.dtype == np.bool

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

        assert subject_mask.dtype == np.bool
        assert bbox_mask.dtype == np.bool

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
            if np.random.rand(1)[0] < self.conf.sample_subject_ratio:
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
                N_patch=self.conf.N_patches,
                ray_mask=ray_mask, 
                subject_mask=subject_mask, 
                bbox_mask=bbox_mask,
                patch_size=self.conf.patch_size, 
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
        for i in range(self.conf.N_patches):
            x_min, y_min = patch_info['xy_min'][i] 
            x_max, y_max = patch_info['xy_max'][i]
            targets.append(image[y_min:y_max, x_min:x_max])
        target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

        patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

        return rays_o, rays_d, rays_img, rays_alpha, near, far, target_patches, patch_masks, patch_div_indices