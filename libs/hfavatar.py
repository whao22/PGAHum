import os
from typing import Optional
import numpy as np
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import lpips
import cv2
from collections import OrderedDict
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn

from libs.utils import pytorch_ssim
from libs.utils.general_utils import augm_rots, sample_sdf_from_grid

from libs.renderers.renderer import IDHRenderer
from libs.renderers.loss import IDHRLoss

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)

def mse2psnr(mse):
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)
    
class HFAvatar(pl.LightningModule):
    def __init__(self, 
                conf, 
                base_exp_dir, 
                pose_decoder,
                motion_basis_computer,
                offset_net,
                non_rigid_mlp,
                nerf_outside,
                deviation_network,
                color_network,
                sdf_network,
                sdf_decoder, 
                skinning_model,
                N_frames):
        super().__init__()
        self.conf = conf
  
        # make exp dir
        if base_exp_dir is not None:
            self.conf.put('general.base_exp_dir', base_exp_dir)
        
        self.base_exp_dir = conf['general.base_exp_dir']
        print(f"Base experiment directory is {self.base_exp_dir}")
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        # flags
        self.anneal_end = conf['train.anneal_end']
        self.total_bones = conf['general.total_bones']
        self.batch_size = conf['train.batch_size']
        self.N_rays = conf.dataset.patch_size**2 * conf.dataset.N_patches
        self.resolution_level = conf.dataset.res_level
        self.lr = conf.train.lr
        self.warm_up_end = conf.train.warm_up_end
        self.view_input_noise = conf.train.view_input_noise
        self.pose_input_noise = conf.train.pose_input_noise
        self.nv_noise_type = conf.train.nv_noise_type
        self.sdf_mode = conf.train.sdf_mode
        self.inner_sampling = conf.dataset.inner_sampling
        
        # Loss Function
        rgb_loss_type = conf['train.rgb_loss_type']
        self.criteria = IDHRLoss(**conf['train.weights'],
                                 rgb_loss_type=rgb_loss_type)
        
        # Networks
        self.pose_decoder = pose_decoder
        self.motion_basis_computer = motion_basis_computer
        self.offset_net = offset_net
        self.non_rigid_mlp = non_rigid_mlp
        self.nerf_outside = nerf_outside
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.sdf_decoder = sdf_decoder
        self.sdf_network = sdf_network
        self.skinning_model = skinning_model
        
        # Renderer
        self.latent = nn.Embedding(N_frames, 128)
        self.renderer = IDHRenderer(pose_decoder=self.pose_decoder,
                                    motion_basis_computer=self.motion_basis_computer,
                                    offset_net=self.offset_net,
                                    skinning_model=self.skinning_model,
                                    non_rigid_mlp=self.non_rigid_mlp,
                                    nerf=self.nerf_outside,
                                    sdf_network=self.sdf_network,
                                    deviation_network=self.deviation_network,
                                    color_network=self.color_network,
                                    total_bones=self.total_bones,
                                    sdf_mode = self.sdf_mode,
                                    inner_sampling = self.inner_sampling,
                                    non_rigid_multries = conf.model.non_rigid.multires,
                                    non_rigid_kick_in_iter= conf.model.non_rigid.kick_in_iter,
                                    non_rigid_full_band_iter=conf.model.non_rigid.full_band_iter,
                                    pose_refine_kick_in_iter=conf.model.pose_refiner.kick_in_iter,
                                    **self.conf.model.neus_renderer)
        
    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.global_step / self.anneal_end])
    
    def get_sdf_decoder(self, inputs, idx):
        rots = inputs['dst_Rs']
        Jtrs = inputs['tjoints']

        batch_size = rots.size(0)
        device = rots.device
        
        decoder_input = {'coords': torch.zeros(1, 1, 3, dtype=torch.float32, device=device), 'rots': rots[0].reshape(-1, 9).unsqueeze(0), 'Jtrs': Jtrs[0].unsqueeze(0)}
        if 'geo_latent_code_idx' in inputs.keys():
            geo_latent_code = self.latent(inputs['geo_latent_code_idx'])
            decoder_input.update({'latent': geo_latent_code})

        # Do augmentation to input poses and views, if applicable
        with_noise = False
        if (self.pose_input_noise or self.view_input_noise) and not eval:
            if np.random.uniform() <= 0.5:
                if self.pose_input_noise:
                    decoder_input['rots_noise'] = torch.normal(mean=0, std=0.1, size=rots.shape, dtype=rots.dtype, device=device)
                    inputs['pose_cond']['rot_noise'] = torch.normal(mean=0, std=0.1, size=(batch_size,9), dtype=rots.dtype, device=device)
                    inputs['pose_cond']['trans_noise'] = torch.normal(mean=0, std=0.1, size=(batch_size,3), dtype=rots.dtype, device=device)

                if self.view_input_noise:
                    if self.nv_noise_type == 'gaussian':
                        inputs['pose_cond']['view_noise'] = torch.normal(mean=0, std=0.1, size=inputs['ray_dirs'].shape, dtype=rots.dtype, device=device)
                    elif self.nv_noise_type == 'rotation':
                        inputs['pose_cond']['view_noise'] = torch.tensor(augm_rots(45, 45, 45), dtype=torch.float32, device=device).unsqueeze(0)
                    else:
                        raise ValueError('wrong nv_noise_type, expected either gaussian or rotation, got {}'.format(self.nv_noise_type))

                with_noise = True
        
        # Generate SDF network from hypernetwork (MetaAvatar)
        output = self.sdf_decoder(decoder_input) 
        # output['model_out'] = output['model_out'] + sample_sdf_from_grid(decoder_input['coords'], inputs['smpl_sdf']) # TODO update sdf
        sdf_decoder = output['decoder']
        sdf_params = output['params']
        return sdf_decoder, sdf_params
    
    def training_step(self, batch, idx):
        sdf_decoder = None
        sdf_params = None
        if self.sdf_mode == 'hyper_net':
            sdf_decoder, sdf_params = self.get_sdf_decoder(batch, idx)
        
        # forward
        render_out = self.renderer.render(batch, self.global_step, cos_anneal_ratio=self.get_cos_anneal_ratio(), sdf_decoder=sdf_decoder)
        loss_results = self.criteria(render_out, batch, sdf_params)
        
        # loss
        self.log('loss_color', loss_results['loss_color'], on_step=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        self.log('loss_mask', loss_results['loss_mask'], on_step=True, logger=True, batch_size=self.batch_size)
        self.log('loss_eikonal', loss_results['loss_eikonal'], on_step=True, logger=True, batch_size=self.batch_size)
        self.log('loss_pips', loss_results['loss_pips'], on_step=True, logger=True, batch_size=self.batch_size)
        self.log('loss_skinning_weights', loss_results['loss_skinning_weights'], on_step=True, logger=True, batch_size=self.batch_size)
        self.log('loss_params', loss_results['loss_params'], on_step=True, logger=True, batch_size=self.batch_size)
        self.log('loss_pose_refine', loss_results['loss_pose_refine'], on_step=True, logger=True, batch_size=self.batch_size)
        
        self.log('loss', loss_results['loss'], on_step=True, logger=True, batch_size=self.batch_size)
        
        return loss_results['loss']
        
    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        self.update_learning_rate(optimizer)
    
    def test_step(self, data, idx):
        sdf_decoder = None
        sdf_params = None
        if self.sdf_mode == 'hyper_net':
            sdf_decoder, sdf_params = self.get_sdf_decoder(data, idx)
        
        def feasible(key):
            return (key in render_out) and (render_out[key] is not None)
        
        with torch.enable_grad():
            n_batches = int(data['batch_rays'].size(1) / self.N_rays)
            out_rgb_fine = []
            out_normal_fine = []

            for i in tqdm(range(n_batches)):
                data_batch = data.copy()
                data_batch['batch_rays'] = data_batch['batch_rays'][:, i*self.N_rays:(i+1)*self.N_rays]
                if self.inner_sampling:
                    data_batch['z_vals'] = data_batch['z_vals'][:, i*self.N_rays:(i+1)*self.N_rays]
                    data_batch['hit_mask'] = data_batch['hit_mask'][:, i*self.N_rays:(i+1)*self.N_rays]
                render_out = self.renderer.render(data_batch, self.global_step, cos_anneal_ratio=self.get_cos_anneal_ratio(), sdf_decoder=sdf_decoder)

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
                torch.cuda.empty_cache()
                del render_out
                del data_batch

            img_fine = None
            H, W, E = data['height'].squeeze(), data['width'].squeeze(), data['extrinsic'].squeeze()
            
            if len(out_rgb_fine) > 0:
                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
                if self.resolution_level == 1:
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

            for i in range(img_fine.shape[-1]):
                if len(out_rgb_fine) > 0:
                    pred_image = img_fine[..., i]
                    
                if len(out_normal_fine) > 0:
                    mormal_map = normal_img[..., i]
            
            # log
            self.logger.log_image(key="test_samples",
                    images=[pred_image, mormal_map],
                    caption=["pred image", "normal map"]
            )
    
    def validation_step(self, data, idx):
        sdf_decoder = None
        sdf_params = None
        if self.sdf_mode == 'hyper_net':
            sdf_decoder, sdf_params = self.get_sdf_decoder(data, idx)
        
        def feasible(key):
            return (key in render_out) and (render_out[key] is not None)
        
        with torch.enable_grad():
            n_batches = int(data['batch_rays'].size(1) / self.N_rays)
            out_rgb_fine = []
            out_normal_fine = []

            for i in tqdm(range(n_batches)):
                data_batch = data.copy()
                data_batch['batch_rays'] = data_batch['batch_rays'][:, i*self.N_rays:(i+1)*self.N_rays]
                if self.inner_sampling:
                    data_batch['z_vals'] = data_batch['z_vals'][:, i*self.N_rays:(i+1)*self.N_rays]
                    data_batch['hit_mask'] = data_batch['hit_mask'][:, i*self.N_rays:(i+1)*self.N_rays]
                render_out = self.renderer.render(data_batch, self.global_step, cos_anneal_ratio=self.get_cos_anneal_ratio(), sdf_decoder=sdf_decoder)

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
                torch.cuda.empty_cache()
                del render_out
                del data_batch

            img_fine = None
            H, W, E = data['height'].squeeze(), data['width'].squeeze(), data['extrinsic'].squeeze()
            img_true = data['batch_rays'][..., 6:9].reshape(H,W,3).squeeze()
            
            if len(out_rgb_fine) > 0:
                img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)
                if self.resolution_level == 1:
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
                
            for i in range(img_fine.shape[-1]):
                if len(out_rgb_fine) > 0:
                    pred_image = img_fine[..., i]
                    gt_image = img_true.detach().cpu().numpy()*255
                    
                if len(out_normal_fine) > 0:
                    mormal_map = normal_img[..., i]
            
            # log
            self.logger.log_image(key="validation_samples",
                    images=[pred_image, gt_image, mormal_map],
                    caption=["pred image", 'GT image',"normal map"]
            )
    
    def update_learning_rate(self, optimizer):
        decay_rate = 0.1
        decay_steps = self.warm_up_end * 1000
        learning_factor = decay_rate ** (self.global_step / decay_steps)
        
        for param_group in optimizer.param_groups:
            if f"lr_{param_group['name']}" in self.conf.train:
                base_lr = self.conf.train[f'lr_{param_group["name"]}']
                new_lr = base_lr * learning_factor
            else:
                new_lr = self.lr * learning_factor
            param_group['lr'] = new_lr
        self.log('lr', new_lr, on_step=True, logger=True, batch_size=self.batch_size)
        
    def configure_optimizers(self):
        # get optimizer
        nets_to_train={
            # 'sdf_network': self.sdf_network,
            'pose_decoder': self.pose_decoder,
            'offset_net': self.offset_net,
            'skinning_model': self.skinning_model,
            'non_rigid_mlp': self.non_rigid_mlp,
            'nerf_outside': self.nerf_outside,
            'deviation_network': self.deviation_network,
            'color_network': self.color_network
        }
        if self.sdf_mode == 'hyper_net':
            nets_to_train['sdf_decoder_net'] = self.sdf_decoder.net.layers
            nets_to_train['sdf_decoder_pose_encoder'] = self.sdf_decoder.pose_encoder
        else:
            nets_to_train['sdf_network'] = self.sdf_network
            
        optimizer = self.get_optimizer_(nets_to_train)
        return optimizer
    
    def get_optimizer_(self, nets_to_train):
        params_to_train = []
        for model in nets_to_train:
            lr = self.conf.train.get_float(f'lr_{model}', 0)
            params_to_train += [{
                "params": nets_to_train[model].parameters(),
                "lr": lr if lr!= 0.0 else self.lr,
                "name": model
            }]
        
        optimizer = torch.optim.Adam(params_to_train, lr=self.lr, betas=(0.9, 0.999))
        return optimizer