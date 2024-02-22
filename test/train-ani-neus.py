import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from libs.datasets.dataset_zjumocap import ZJUMoCapDataset
from libs.models.fields_high import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from libs.utils.network_utils import MotionBasisComputer
from libs.models.deconv_vol_decoder import MotionWeightVolumeDecoder
from libs.models.nonrigid import NonRigidMotionMLP
from libs.models.pose_refine import BodyPoseRefiner
from libs.renderers.renderer_high_ani_neus import NeuSRenderer

import json
import lpips as lpips_lib



import matplotlib
matplotlib.use('Agg')

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

def mse2psnr(mse):
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)

class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False, ckpt_name=None, base_exp_dir=None, end_iter=None):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        if base_exp_dir is not None:
            self.base_exp_dir = base_exp_dir
            self.conf.put('general.base_exp_dir', base_exp_dir)
        else:
            self.base_exp_dir = self.conf['general.base_exp_dir']
        print(f"Base experiment directory is {self.base_exp_dir}")
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = ZJUMoCapDataset(self.conf['dataset'])
        ################################################
        ###   DEBUG SENTENCES
        ################################################
        # self.dataset.gen_random_rays_at(0, 512)
        # def _worker_init_fn(worker_id):
        #     np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))
        # dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=1,
        #     shuffle=True,
        #     num_workers=1,
        #     worker_init_fn=_worker_init_fn,
        #     generator=torch.Generator(device='cuda'),
        # )
        # for data in dataloader:
        #     print("aaa")
        # self.train_images = self.dataset.images.cuda()
        
        self.iter_step = 0

        # Training parameters
        if end_iter is not None:
            self.end_iter = end_iter
            self.conf.put('train.end_iter', end_iter)
        else:
            self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('general.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.ckpt_name = ckpt_name
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        ## HumanNerf
        self.pose_decoder = BodyPoseRefiner(total_bones=self.conf.general.total_bones, **self.conf['model.pose_refiner']).to(self.device)
        self.motion_basis_computer = MotionBasisComputer(self.conf.general.total_bones).to(self.device) # no embedding
        self.mweight_vol_decoder = MotionWeightVolumeDecoder(embedding_size=self.conf.model.mweight_volume.embedding_size,
                                                             volume_size=self.conf.model.mweight_volume.volume_size,
                                                             total_bones=self.conf.general.total_bones
                                                             ).to(self.device) # const embedding
        self.non_rigid_mlp = NonRigidMotionMLP(condition_code_size=self.conf.model.non_rigid.condition_code_size,
                                            mlp_depth=self.conf.model.non_rigid.mlp_depth,
                                            mlp_width=self.conf.model.non_rigid.mlp_width,
                                            multires=self.conf.model.non_rigid.multires,
                                            i_embed=self.conf.model.non_rigid.i_embed,
                                            skips=self.conf.model.non_rigid.skips,
                                            kick_in_iter=self.conf.model.non_rigid.kick_in_iter,
                                            full_band_iter=self.conf.model.non_rigid.full_band_iter
                                            ).to(self.device) # hannw embedding
        ## HF-Neus
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        self.lpips_vgg_fn = lpips_lib.LPIPS(net='vgg').to(self.device)  # net="alex"

        # get optimizer
        nets_to_train={}
        nets_to_train['pose_decoder']=self.pose_decoder
        nets_to_train['mweight_vol_decoder']=self.mweight_vol_decoder
        nets_to_train['non_rigid_mlp']=self.non_rigid_mlp
        nets_to_train['nerf_outside']=self.nerf_outside
        nets_to_train['sdf_network']=self.sdf_network
        nets_to_train['deviation_network']=self.deviation_network
        nets_to_train['color_network']=self.color_network
        self.optimizer = self.get_optimizer(nets_to_train)

        self.renderer = NeuSRenderer(self.pose_decoder,
                                     self.motion_basis_computer,
                                     self.mweight_vol_decoder,
                                     self.non_rigid_mlp,
                                     self.nerf_outside,
                                     self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if self.ckpt_name is not None:
                latest_model_name = self.ckpt_name
            else:
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                num = -1
                for model_name in model_list_raw:
                    iter = int(model_name[5:-4])
                    if model_name[-3:] == 'pth' and iter <= self.end_iter:
                        if iter > num:
                            num = iter
                            model_list = model_name
                # model_list.sort()
                latest_model_name = model_list  # [-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train' and not is_continue:
            self.file_backup()

    def get_optimizer(self, nets_to_train):
        params_to_train = []
        for model in nets_to_train:
            lr = self.conf.train.get_float(f'lr_{model}', 0)
            params_to_train += [{
                "params": nets_to_train[model].parameters(),
                "lr": lr if  lr!= 0.0 else self.conf.train.learning_rate,
                "name": model
            }]
        
        optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate, betas=(0.9, 0.999))
        return optimizer
    
    
    def data_cpu_to_gpu(self, data_cpu, exclude_keys=['frame_name', 'patch_div_indices']):
        if exclude_keys is None:
            exclude_keys = []

        data_gpu = {}
        for key, val in data_cpu.items():
            if key in exclude_keys:
                data_gpu[key]=val
                continue

            if isinstance(val, np.ndarray):
                data_gpu[key]=torch.from_numpy(val).float().cuda()
            elif isinstance(val, int):
                data_gpu[key]=torch.tensor(val, dtype=torch.int64).cuda()
            else:
                data_gpu[key]=val

        return data_gpu
    
    def get_loss(self, predicted, target):
        def unpack_image(predicted, target):
            bg_color = target['background_color']
            div_indices = target['patch_div_indices']
            patch_masks = target['patch_masks']
            target_patches = target['target_patches']
            N_patch = len(div_indices) - 1
            pre_color = predicted['color']
            
            assert patch_masks.shape[0] == N_patch
            assert target_patches.shape[0] == N_patch

            patch_imgs = bg_color.expand(target_patches.shape).clone() # (N_patch, H, W, 3)
            for i in range(N_patch):
                patch_imgs[i, patch_masks[i].bool()] = pre_color[div_indices[i]:div_indices[i+1]]

            return patch_imgs, target_patches
            
        def get_color_rebuild_loss(loss_names, loss_weights, pre, tar):
            loss = 0
            if "mse" in loss_names:
                loss += img2mse(pre, tar) * loss_weights['mse']
            if "l2" in loss_names:
                loss += img2l1(pre, tar) * loss_weights['l2']
            if "lpips" in loss_names:
                loss += torch.mean(self.lpips_vgg_fn(pre.permute(0,3,1,2), tar.permute(0,3,1,2))) * loss_weights['lpips']
            return loss
        
        # loss_color
        color_weights = self.conf.train.color_weights
        loss_names = list(color_weights.keys())
        pre_color, tar_color = unpack_image(predicted, target)
        loss_color = get_color_rebuild_loss(loss_names, color_weights, pre_color*target['patch_masks'][...,None], tar_color)
        
        # loss_mask
        tar_mask = target['batch_rays'][..., 9:10]
        pre_mask = torch.sum(predicted['weights'], dim=-1, keepdim=True)
        loss_mask = F.binary_cross_entropy(pre_mask.clip(1e-3, 1.0-1e-3), tar_mask)
        
        # loss_eikonal
        loss_eikonal = predicted['gradient_error']
        
        results={
            'loss_color': loss_color,
            'loss_mask': loss_mask,
            'loss_eikonal': loss_eikonal
        }
        return results
    

    def train(self):
        # lpips_vgg_fn = self.lpips_vgg_fn
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        image_perm = self.get_image_perm()
        res_step = self.end_iter - self.iter_step

        progress_data = 1.0
        self.nerf_outside.progress.data.fill_(progress_data)

        for iter_i in tqdm(range(res_step)):
            self.optimizer.zero_grad()
            
            # get random batch data
            idx_list = image_perm[self.iter_step % len(image_perm)]
            data = self.dataset.gen_random_rays_at(idx_list, self.batch_size)
            data = self.data_cpu_to_gpu(data, exclude_keys=['frame_name', 'patch_div_indices'])
            
            # forward
            render_out = self.renderer.render(data, self.iter_step, self.conf, cos_anneal_ratio=self.get_cos_anneal_ratio())

            # calculate losses
            results = self.get_loss(render_out, data)
            loss_color = results['loss_color']
            loss_mask = results['loss_mask']
            loss_eikonal = results['loss_eikonal']
            loss = loss_color + loss_mask * self.mask_weight + loss_eikonal * self.igr_weight
            
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/loss_color', loss_color, self.iter_step)
            self.writer.add_scalar('Loss/loss_mask', loss_mask, self.iter_step)
            self.writer.add_scalar('Loss/loss_eikonal', loss_eikonal, self.iter_step)


            if self.iter_step % self.report_freq == 0 or self.iter_step == 1:
                print(self.base_exp_dir)
                outstr = 'iter:{:8>d} loss = {} lr={}\n'.format(self.iter_step, loss,
                                                                self.optimizer.param_groups[0]['lr'])
                print(outstr)
                f = os.path.join(self.base_exp_dir, 'logs', 'loss.txt')
                with open(f, 'a') as file:
                    file.write(outstr)

            if self.iter_step % self.save_freq == 0 or self.iter_step == 1:
                self.save_checkpoint()

            
            if self.iter_step % self.val_freq == 0 or self.iter_step == 1:
                if self.iter_step <= 10000:
                    self.validate_image()
                elif self.iter_step % (self.val_freq * 5) == 0:
                    self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0 or self.iter_step == 1:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step > self.end_iter / 2:
                progress_data = 1.0
            else:
                progress_data = 0.5 + self.iter_step / (self.end_iter)
            self.pose_decoder.progress.data.fill_(progress_data)
            self.motion_basis_computer.progress.data.fill_(progress_data)
            self.mweight_vol_decoder.progress.data.fill_(progress_data)
            self.non_rigid_mlp.progress.data.fill_(progress_data)
            self.sdf_network.progress.data.fill_(progress_data)
            self.color_network.progress.data.fill_(progress_data)


            if (iter_i+1) % len(image_perm) == 0:
                image_perm = self.get_image_perm()


    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)


    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])


    def update_learning_rate(self):
        decay_rate = 0.1
        decay_steps = self.warm_up_end * 1000
        learning_factor = decay_rate ** (self.iter_step / decay_steps)
        
        for param_group in self.optimizer.param_groups:
            if f"lr_{param_group['name']}" in self.conf.train:
                base_lr = self.conf.train[f'lr_{param_group["name"]}']
                new_lr = base_lr * learning_factor
            else:
                new_lr = self.learning_rate * learning_factor
            param_group['lr'] = new_lr


    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))


    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name),
                                map_location=self.device)
        self.pose_decoder.load_state_dict(checkpoint['pose_decoder'])
        self.motion_basis_computer.load_state_dict(checkpoint['motion_basis_computer'])
        self.mweight_vol_decoder.load_state_dict(checkpoint['mweight_vol_decoder'])
        self.non_rigid_mlp.load_state_dict(checkpoint['non_rigid_mlp'])
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')


    def save_checkpoint(self):
        checkpoint = {
            'pose_decoder': self.pose_decoder.state_dict(),
            'motion_basis_computer': self.motion_basis_computer.state_dict(),
            'mweight_vol_decoder': self.mweight_vol_decoder.state_dict(),
            'non_rigid_mlp': self.non_rigid_mlp.state_dict(),
            'nerf': self.nerf_outside.state_dict(),
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint,
                   os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))


    def validate_image(self, idx=-1, resolution_level=-1):
        lpips_vgg_fn = self.lpips_vgg_fn
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        outstr = 'Validate: iter: {}, camera: {}\n'.format(self.iter_step, idx)
        print(outstr)
        if resolution_level == 1:
            f = os.path.join(self.base_exp_dir, 'logs', 'image_metric.txt')
            with open(f, 'a') as file:
                file.write(outstr)

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level


        data = self.dataset.gen_rays_at(idx, res_level=resolution_level)
        data = self.data_cpu_to_gpu(data, exclude_keys=['frame_name'])
        
        n_batches = int(data['batch_rays'].size(0) / self.batch_size)
        
        out_rgb_fine = []
        out_normal_fine = []
        print('Rendering image...')
        for i in tqdm(range(n_batches)):
            data_batch = data.copy()
            data_batch['batch_rays'] = data_batch['batch_rays'][i*self.batch_size:(i+1)*self.batch_size]
            render_out = self.renderer.render(data_batch, self.iter_step, self.conf, cos_anneal_ratio=self.get_cos_anneal_ratio())

            def feasible(key):
                return (key in render_out) and (render_out[key] is not None)

            if feasible('color'):
                out_rgb_fine.append(render_out['color'].detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                n_samples = self.renderer.n_samples + self.renderer.n_importance
                normals = render_out['gradients'] * render_out['weights'][:, :n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                # normals = (normals.sum(dim=1)**2).sum(dim=1,keepdim=True).sqrt().tile(1,3).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out
            del data_batch

        img_fine = None
        H, W, E = data['height'], data['width'], data['extrinsic']
        img_true = data['batch_rays'][..., 6:9].reshape(H,W,3)
        
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
            if resolution_level == 1:
                color_fine = torch.from_numpy(np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3])).to(self.device)
                true_rgb = self.dataset.images[idx].to(self.device)
                color_error = color_fine - true_rgb
                color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='mean')
                outstr = '{0:4d} img: loss: {1:.2f}\n'.format(idx, color_fine_loss)
                print(outstr)
                f = os.path.join(self.base_exp_dir, 'logs', 'image_metric.txt')
                with open(f, 'a') as file:
                    file.write(outstr)
                mse = F.mse_loss(color_fine, true_rgb).item()
                psnr = mse2psnr(mse)
                ssim = pytorch_ssim.ssim(color_fine.permute(2, 0, 1).unsqueeze(0), true_rgb.permute(2, 0, 1).unsqueeze(0)).item()
                lpips_loss = lpips_vgg_fn(color_fine.permute(2, 0, 1).unsqueeze(0).contiguous(),
                                          true_rgb.permute(2, 0, 1).unsqueeze(0).contiguous(), normalize=True).item()
                outstr = '{0:4d} img: PSNR: {1:.2f}, SSIM: {2:.2f}, LPIPS {3:.2f}\n'.format(idx, psnr, ssim, lpips_loss)
                print(outstr)
                f = os.path.join(self.base_exp_dir, 'logs', 'image_metric.txt')
                with open(f, 'a') as file:
                    file.write(outstr)


        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(E[:3,:3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)
            # maxv = normal_img.max()
            # normal_img = (normal_img[:, :, None]
            #               .reshape([H, W, 3, -1]) / maxv * 256).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i], img_true.detach().cpu().numpy()*255])[...,::-1])
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])


    def render_novel_image(self, idx_0, idx_1, ratio, resolution_level):
        """
        Interpolate view between two cameras.
        """

        rays_o, rays_d = self.dataset.gen_rays_between(idx_0, idx_1, ratio, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)

        out_rgb_fine = []
        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            background_rgb = torch.ones([1, 3]) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              background_rgb=background_rgb)

            out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3]) * 256).clip(0, 255).astype(np.uint8)
        return img_fine


    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        print('Validate: iter: {} mesh'.format(self.iter_step))
        bound_min = torch.tensor(self.dataset.canonical_bbox['min_xyz'], dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.canonical_bbox['max_xyz'], dtype=torch.float32)

        vertices, triangles = \
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        # if world_space:
        #     vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]


        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        logging.info('End')


    def interpolate_view(self, img_idx_0, img_idx_1):
        images = []
        n_frames = 60
        for i in range(n_frames):
            print(i)
            images.append(self.render_novel_image(img_idx_0,
                                                  img_idx_1,
                                                  np.sin(((i / n_frames) - 0.5) * np.pi) * 0.5 + 0.5,
                                                  resolution_level=4))
        for i in range(n_frames):
            images.append(images[n_frames - i - 1])

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_dir = os.path.join(self.base_exp_dir, 'render')
        os.makedirs(video_dir, exist_ok=True)
        h, w, _ = images[0].shape
        writer = cv.VideoWriter(os.path.join(video_dir,
                                             '{:0>8d}_{}_{}.mp4'.format(self.iter_step, img_idx_0, img_idx_1)),
                                fourcc, 30, (w, h))

        for image in images:
            writer.write(image)

        writer.release()


if __name__ == '__main__':

    print('Hello Wooden')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--ckpt_name', type=str, default=None)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--image_idx', type=int, default=0)
    parser.add_argument('--image_resolution', type=int, default=4)
    parser.add_argument('--mesh_resolution', type=int, default=512)
    parser.add_argument('--base_exp_dir', type=str, default=None)
    parser.add_argument('--end_iter', type=int, default=None)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.ckpt_name, args.base_exp_dir, args.end_iter)

    if args.mode == 'train':
        runner.train()
        runner.validate_mesh(world_space=True, resolution=args.mesh_resolution,
                             threshold=args.mcube_threshold)  # world_space=True
        runner.validate_image(idx=args.image_idx, resolution_level=args.image_resolution)
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=args.mesh_resolution, threshold=args.mcube_threshold) # world_space=True
    elif args.mode == 'validate_image':
        runner.validate_image(idx=args.image_idx, resolution_level=args.image_resolution)
    elif args.mode.startswith('interpolate'):  # Interpolate views given two image indices
        _, img_idx_0, img_idx_1 = args.mode.split('_')
        img_idx_0 = int(img_idx_0)
        img_idx_1 = int(img_idx_1)
        runner.interpolate_view(img_idx_0, img_idx_1)
