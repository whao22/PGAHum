import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.utils.data
import trimesh

from libs.utils.general_utils import rays_mesh_intersections_pcu, get_02v_bone_transforms
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


class ZJUMoCapDataset_ODP(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_folder='',
                 subjects=['CoreView_313'],
                 new_pose_mode='zjumocap',
                 new_pose_folder='data/data_prepared/CoreView_392',
                 mode='test',
                 resize_img_scale=1.,
                 start_frame=0,
                 end_frame=-1,
                 sampling_rate=1,
                 views=[],
                 box_margin=0.05,
                 ray_shoot_mode='default',
                 backgroung_color=None,
                 N_samples=64,
                 inner_sampling=False,
                 res_level=4):
        assert len(subjects) == 1, 'SINGLE PERSON! Please make sure the length of subjects list is one.'
        assert len(views) == 1, 'MONOCULAR! Please make sure the length of views list is one.'
        
        # meta data
        self.mode = mode
        self.N_samples = N_samples
        self.inner_sampling = inner_sampling
        self.res_level = res_level
        self.bgcolor = np.array(backgroung_color, dtype=np.float32) if backgroung_color is not None else (np.random.rand(3) * 255).astype(np.float32)
        self.ray_shoot_mode = ray_shoot_mode
        self.bbox_offset=box_margin
        self.resize_img_scale=resize_img_scale
        self.new_pose_mode = new_pose_mode
        self.new_pose_folder = new_pose_folder
        
        self.dataset_path = os.path.join(dataset_folder, subjects[0])
        self.camera_info_dir = os.path.join(self.dataset_path, views[0])
        
        # load data
        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()
        self.cnl_gtfms = get_canonical_global_tfms(self.canonical_joints) # (24, 4, 4)
        self.gtfs_02v = get_02v_bone_transforms(self.canonical_joints)
        self.cameras = self.load_train_cameras()
        self.mesh_infos, self.framelist = self.load_new_mesh_infos()
        self.framelist = self.framelist[start_frame:end_frame:sampling_rate]
        
        self.faces = np.load('data/body_models/misc/faces.npz')['faces']
        self.skinning_weights = dict(np.load('data/body_models/misc/skinning_weights_all.npz'))
        self.posedirs = dict(np.load('data/body_models/misc/posedirs_all.npz'))
        self.J_regressor = dict(np.load('data/body_models/misc/J_regressors.npz'))
        self.smpl_sdf = np.load(os.path.join(self.dataset_path, 'smpl_sdf.npy'), allow_pickle=True).item()
        
        print(f'[Dataset Path]: {self.dataset_path} / {self.mode}. -- Total Frames: {self.__len__()}')
        
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
        cameras = None
        with open(os.path.join(self.camera_info_dir, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras

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

    def load_new_mesh_infos(self):
        mesh_infos = None
        if self.new_pose_mode == 'zjumocap':
            with open(os.path.join(self.new_pose_folder, 'mesh_infos.pkl'), 'rb') as f: 
                mesh_infos = pickle.load(f)      
        else:
            raise ValueError(f"Unimplement pose mode!")
        
        framelist = []
        for frame_name in mesh_infos.keys():
            framelist.append(frame_name)
            # bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['posed_vertices'])
            mesh_infos[frame_name]['bbox'] = bbox

        return mesh_infos, framelist
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.canonical_joints.astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32'),
            'posed_vertices': self.mesh_infos[frame_name]['posed_vertices'].astype('float32')
        }

    def sample_image_rays(self, res_level, rays_o, rays_d, near, far):
        rays_o = rays_o[::res_level, ::res_level].reshape(-1, 3) # (N_rays, 3)
        rays_d = rays_d[::res_level, ::res_level].reshape(-1, 3) # (N_rays, 3)
        near = near[::res_level, ::res_level].reshape(-1, 1) # (N_rays, 1)
        far = far[::res_level, ::res_level].reshape(-1, 1) # (N_rays, 1)
        
        return rays_o, rays_d, near, far

    def gen_rays_for_infer(self, idx):
        res_level = self.res_level
        frame_name = self.framelist[idx]
        H, W = 1024, 1024 # zjumocap image height and width

        # dst skeleton info
        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_bbox = dst_skel_info['bbox'] 
        dst_poses = dst_skel_info['poses'] # (24 x 3,)
        dst_tpose_joints = dst_skel_info['dst_tpose_joints'] # (24, 3)
        dst_vertices = dst_skel_info['posed_vertices']
        dst_Rs, dst_Ts = body_pose_to_body_RTs(dst_poses, dst_tpose_joints) # (24, 3, 3) (24, 3)
        
        # calculate K, E, T
        assert frame_name in self.cameras
        K = self.cameras[frame_name]['intrinsics'][:3, :3].copy()
        K[:2] *= self.resize_img_scale
        E = self.cameras[frame_name]['extrinsics'].astype(np.float32)
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
        rays_o, rays_d, near, far = self.sample_image_rays(res_level=res_level,
                            rays_o=rays_o,
                            rays_d=rays_d,
                            near=near.reshape(H, W, 1),
                            far=far.reshape(H, W, 1)) # (N_rays, 12) denote there are N_rays rays
        batch_rays = np.concatenate([rays_o, rays_d, np.zeros_like(rays_o), np.zeros_like(rays_o), near, far], axis=-1) # (N_rays, 12), Columns 6-11 are just for space and to avoid code changes
        
        # return result
        min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
        max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
        results={
            'batch_rays': batch_rays.astype(np.float32), # ndarray (N_rays, 12)
            
            'frame_name': frame_name, # str
            'extrinsic': E.astype(np.float32), # ndarray (4, 4)
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
            'geo_latent_code_idx': idx
            # 'rots': pose_rot.astype(np.float32), # (24, 9), pose rotation, where the root rotation is identity
            # 'Jtrs': Jtr_norm.astype(np.float32), # (24 3), T-pose joint points
        }
        if self.inner_sampling:
            z_vals, inter_mask = self.get_z_vals(near, far, rays_o, rays_d, dst_vertices, self.faces)
            results['z_vals'] = z_vals.astype(np.float32)
            results['hit_mask'] = inter_mask
        return results

    def __len__(self):
        return len(self.framelist)
    
    def __getitem__(self, idx):
        if self.mode in ['test','val']:
            return self.gen_rays_for_infer(idx)
        else:
            raise ValueError('Only test mode is supported!')
    
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
    
    def get_z_vals(self, near, far, rays_o, rays_d, vertices, faces):
        """Get z-depth w.r.t. z_vals for sampling points.

        Args:
            near (ndarray): (N_rays, 1)
            far (ndarray): (N_rays, 1)
            rays_o (ndarray): (N_rays, 3), the start points of rays
            rays_d (ndarray): (N_rays, 3), the direction of rays
            vertices (ndarray): (N_vertices, 3)
            faces (ndarray): (N_faces, 3)

        Returns:
            z_vals (ndarray): (N_rays, N_samples), z_vals for depth of sampling points in rays
            inter_mask (ndarray): (N_rays,), mask with any intersection, 
        """
        ALPHA = 0.008 * 0.5 # 0.008 is determined by calculating the mean length of all edge
        rays_o = rays_o + rays_d * near
        num_intersections = np.zeros_like(rays_o[..., :1], dtype=np.int8) # the number of intersections between rays and mesh, where the start points is the rays_o 
        dist_lst = [near] # maintain distances of current rays_o and current intersection.
        hit_mask = near < np.inf
        mask_lst = [hit_mask]
        
        # ray mesh intersections computing
        n_iter = 0
        while np.any(hit_mask):
            hit_mask, _, hit_points, cur_dists = \
                rays_mesh_intersections_pcu(rays_o=rays_o, rays_d=rays_d, vertices=vertices, faces=faces)
            
            num_intersections[hit_mask] = num_intersections[hit_mask] + 1
            rays_o = rays_o + rays_d * (cur_dists[..., None] + ALPHA)
            if n_iter == 0:
                dist_lst.append(cur_dists[..., None])
            else:
                dist_lst.append(cur_dists[..., None] + ALPHA)
            mask_lst.append(hit_mask[..., None])
            n_iter += 1
        
        # for an interval of intersection, calculate a mask.
        masks = np.concatenate(mask_lst, axis=-1)
        hit_masks = masks[:, 1:-1]                                              # first column all True, last column all False, so drop them.
        if hit_masks.size == 0 or hit_masks.shape[-1] == 1:
            hit_masks = masks[:, :1]
        else:
            hit_masks = hit_masks[:, :hit_masks.shape[-1]//2*2]
            hit_masks[:, 0::2] = hit_masks[:, 0::2] & hit_masks[:, 1::2]
            hit_masks[:, 1::2] = hit_masks[:, 0::2]                             # (N_rays, N_interval * 2)
            
        # calculate the distance of intersections and camera location
        dists = np.concatenate(dist_lst, axis=-1)
        hit_dists = dists[:, 1:-1]
        if hit_dists.size == 0 or hit_dists.shape[-1] == 1:
            hit_dists = dists[:, :1]                                            # if donot intersect with mesh, assign the dist_intersections to near
        else:
            hit_dists = hit_dists[:, :hit_dists.shape[-1]//2*2]
            hit_dists[hit_dists==np.inf] = 0
            hit_dists = hit_dists * hit_masks                                   # make hit_dists and hit_masks keep identity
            hit_dists = np.cumsum(hit_dists, axis=-1)
            hit_dists = hit_dists + dists[:, :1]                                # (N_rays, N_interval * 2)
        
        z_vals, inter_mask = self.calculate_inner_smpl_depth(near, far, hit_masks, hit_dists)
        return z_vals, inter_mask
    
    def calculate_inner_smpl_depth(self, near, far, hit_masks, hit_dists):
        """Calculate sampling depth in rays given the hit masks and hit distances.

        Args:
            near (ndarray): (N_rays, 1).
            far (ndarray): (N_rays, 1).
            hit_masks (ndarray): (N_rays, N_intervals * 2), each column represents the hit mask of rays and mesh, 
                                N_intervals denotes the number interval of rays and mesh.
            hit_dists (ndarray): (N_rays, N_intervals * 2), each column represents the hit distance between camera 
                                location and intersection.

        Returns:
            z_vals (ndarray): (N_rays, N_samples), z_vals for depth of sampling points in rays
            inter_mask (ndarray): (N_rays,), mask with any intersection, 
        """
        N_rays = len(near)
        N_samples = self.N_samples
        N_interval = hit_masks.shape[-1] // 2
        inter_mask = hit_masks.sum(-1) > 0 # ndarray (N_rays,), mask with any intersection, 
        
        # if N_interval equal with 0, it means no intersection of rays and mesh, 
        # so return the uniform sampling in [near, far].
        if N_interval == 0:
            t = np.linspace(0, 1, N_samples)
            z_vals = near[:, None] + (far[:, None] - near[:, None]) * t[None, :]
            return z_vals, inter_mask
        
        # For interval of intersection, calculate number of points to sample. If value is 0, 
        # it means should sampling in [near, far].
        interval_lst = [hit_dists[:, 2*i+1] - hit_dists[:, 2*i] for i in range(N_interval)]
        interval_lst = np.stack(interval_lst, axis=-1)
        num_sampled = interval_lst / (interval_lst.sum(-1, keepdims=True)+1e-9) * N_samples  # num_sampled denote the number of points should be sampled in intersection interval, 
        num_sampled_cumsum = np.round(num_sampled.cumsum(-1)).astype(np.int32)               # num_sampled is a (N_rays, N_interval) ndarray
        num_sampled = num_sampled_cumsum.copy().astype(np.int32)
        num_sampled[:, 1:] = num_sampled[:, 1:] - num_sampled[:, :-1]
        
        # construct the z_vals for sampleing points in rays
        z_vals = np.zeros([N_rays, N_samples]) # (N_rays, N_samples)
        
        # Deprecated Method. aggsign the z_vals of non-intersectd rays as [near, far]
        # If ray intersect with mesh, update near and far by intersection distance, else donot change.
        t = np.linspace(0, 1, N_samples)
        z_vals[~inter_mask] = near[~inter_mask] + (far[~inter_mask] - near[~inter_mask]) * t[None, :]
        
        # construct z_vals
        grid_idx, _ = np.meshgrid(np.arange(N_samples), np.arange(N_rays))
        for i in range(N_interval):
            # calculated mean distance of current near and current far by num_sampled 
            cur_near = hit_dists[:, 2*i]
            cur_far = hit_dists[:, 2*i+1]
            mean_dist = (cur_far - cur_near) / (num_sampled[:, i]+1e-10)
            mean_dist = np.tile(mean_dist[:, None], [1, N_samples])
            
            # assign the z_vals of intersected rays as (cur_near, cur_far)
            z_vals_tmp = np.zeros([N_rays, N_samples])
            if i==0:
                mask = grid_idx < np.tile(num_sampled_cumsum[:, i:i+1], [1, N_samples])
            else:
                mask1 = grid_idx >= np.tile(num_sampled_cumsum[:, i-1:i], [1, N_samples])
                mask2 = grid_idx < np.tile(num_sampled_cumsum[:, i:i+1], [1, N_samples])
                mask = mask1 & mask2
            z_vals_tmp[mask] = mean_dist[mask]
            if i == 0:
                z_vals_tmp[:, 0] = cur_near
            else:
                idx = num_sampled[:, i-1] -1
                idx[idx==-1] = 0
                z_vals_tmp[np.arange(len(idx)), idx] = cur_near
            z_vals[mask] = np.cumsum(z_vals_tmp, axis=-1)[mask]
        
        return z_vals, inter_mask
