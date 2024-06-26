import os
import torch
from collections import OrderedDict

from libs.datasets.zjumocap_odp import ZJUMoCapDataset_ODP
from libs.datasets.aistplusplus_odp import AISTPlusPlusDataset_ODP
from libs.datasets.zjumocap_mvs import ZJUMoCapDataset_MVS
from libs.datasets.people_snapshot import PeopleSnapshotDataset
from libs.datasets.selfrecon_synthesis import SelfreconSynthesisDataset
from libs.datasets.synthetic_human import SyntheticHumanDataset
from libs.datasets.monocap import MONOCAP
from libs.datasets.h36m import H36M
from libs.hfavatar import HFAvatar
from libs.models.siren_modules import HyperBVPNet
from libs.models.deformer import Deformer
from libs.models.network import ImplicitNetwork

from libs.embeders.hannw_fourier import get_embedder
from libs.models.pose_refine import BodyPoseRefiner
from libs.models.fields_high import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from libs.models.nonrigid import NonRigidMotionMLP
from libs.models.triplane_pet import TriPlaneGenerator

from libs.models.offset import Offset
from libs.utils.network_utils import MotionBasisComputer

# Model
def get_sdf_decoder(cfg, init_weights=True):
    ''' Returns a SDF decoder instance.

    Args:
        cfg (yaml config): yaml config object
        init_weights (bool): whether to initialize the weights for the SDF network with pre-trained model (MetaAvatar)
    '''
    decoder = HyperBVPNet(**cfg['model']['decoder_kwargs'])#.to(device)
    optim_geometry_net_path = cfg['model']["decoder_kwargs"]['geometry_net']
    
    if init_weights and optim_geometry_net_path is not None:
        ckpt = torch.load(optim_geometry_net_path, map_location='cpu')

        decoder_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            if k.startswith('module'):
                k = k[7:]

            if k.startswith('decoder'):
                decoder_state_dict[k[8:]] = v

        decoder.load_state_dict(decoder_state_dict, strict=False)

    return decoder

def get_skinning_model(conf, init_weight=True):
    # skinning_model = ImplicitNetwork(**conf['model.skinning_model'])
    skinning_model = Deformer(**conf['model.skinning_model'])
    
    # init skinning model
    optim_skinning_net_path = conf.get("model.skinning_model.optim_skinning_net_path", None)
    if init_weight and optim_skinning_net_path is not None:
        ckpt = torch.load(optim_skinning_net_path, map_location='cpu')

        skinning_decoder_fwd_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            if k.startswith('module'):
                k = k[7:]

            if k.startswith('skinning_decoder_fwd'):
                skinning_decoder_fwd_state_dict[k[21:]] = v

        skinning_model.load_state_dict(skinning_decoder_fwd_state_dict, strict=False)
    return skinning_model

def load_init_weight(net_path, pose_decoder: torch.nn.Module, non_rigid_mlp: torch.nn.Module):
    ckpt = torch.load(net_path, map_location='cpu')
    
    # pose_decoder
    pose_decoder_fwd_state_dict = OrderedDict()
    for k, v in ckpt['network'].items():
        if k.startswith('module'):
            k = k[7:]

        if k.startswith('pose_decoder'):
            pose_decoder_fwd_state_dict[k[13:]] = v
    pose_decoder.load_state_dict(pose_decoder_fwd_state_dict, strict=False)
    
    # non_rigid_mlp
    non_rigid_mlp_fwd_state_dict = OrderedDict()
    for k, v in ckpt['network'].items():
        if k.startswith('module'):
            k = k[7:]

        if k.startswith('non_rigid_mlp'):
            non_rigid_mlp_fwd_state_dict[k[14:]] = v
    non_rigid_mlp.load_state_dict(non_rigid_mlp_fwd_state_dict, strict=False)
    
    return pose_decoder, non_rigid_mlp
    
def get_model(conf):
    init_weight = conf['model.init_weight']
    total_bones = conf['general.total_bones']
    deform_mode = conf['model.deform_mode']
    sdf_mode = conf['model.sdf_mode']
    net_path = conf['model.nets_path']
    n_outside = conf['model.neus_renderer.n_outside']
    
    models = {}
    # Deformer
    if deform_mode == 0:
        offset_net = Offset(**conf['model.offset_net'])
        _, non_rigid_pos_embed_size = get_embedder(multires=conf.model.non_rigid.multires, 
                                               iter_val=conf.model.non_rigid.i_embed,
                                               kick_in_iter=conf.model.non_rigid.kick_in_iter,
                                               full_band_iter=conf.model.non_rigid.full_band_iter)
        non_rigid_mlp = NonRigidMotionMLP(**conf['model.non_rigid'], pos_embed_size=non_rigid_pos_embed_size)
        models.update({'offset_net': offset_net, 'non_rigid_mlp': non_rigid_mlp})
    elif deform_mode == 1:
        skinning_model = get_skinning_model(conf, init_weight=init_weight)
        models.update({'skinning_model': skinning_model})
    else:
        raise ValueError(f"Deform mode {deform_mode} is not supported!")
    
    # NeRF for outside rendering
    if n_outside > 0:
        nerf_outside = NeRF(**conf['model.nerf'])
        models.update({'nerf_outside': nerf_outside})
    
    # SDF network
    if sdf_mode == 0:
        sdf_network = SDFNetwork(**conf['model.sdf_network'])
        models.update({'sdf_network': sdf_network})
    elif sdf_mode == 1:
        sdf_network = TriPlaneGenerator(**conf['model.triplane_network'])
        models.update({'sdf_network': sdf_network})
    elif sdf_mode == 2:
        sdf_decoder = get_sdf_decoder(conf, init_weights=init_weight)
        models.update({'sdf_decoder': sdf_decoder})
    else:
        raise ValueError(f"SDF mode {sdf_mode} is not supported!")
    
    # Pose refiner and motion basis computer
    pose_decoder = BodyPoseRefiner(total_bones=total_bones, **conf['model.pose_refiner'])
    motion_basis_computer = MotionBasisComputer(total_bones)
    if init_weight and net_path is not None:
        pose_decoder, non_rigid_mlp = load_init_weight(net_path, pose_decoder, non_rigid_mlp)
    
    # Variance and rendering networks for neus rendering
    deviation_network = SingleVarianceNetwork(**conf['model.variance_network'])
    color_network = RenderingNetwork(**conf['model.rendering_network'])        
    models.update({
        'pose_decoder': pose_decoder,
        'motion_basis_computer': motion_basis_computer,
        'deviation_network': deviation_network,
        'color_network': color_network,
    })
    
    return HFAvatar(conf=conf,
                    models=models)

# Datasets
def get_dataset(mode, cfg, view_split=None, subsampling_rate=None, start_frame=None, end_frame=None):
    ''' Returns the dataset.

    Args:
        mode (str): which mode the dataset is. Can be either train, val or test
        cfg (dict): config dictionary
        view_split (list of str): which view(s) to use. If None, will load all views
        subsampling_rate (int): frame subsampling rate for the dataset
        start_frame (int): starting frame
        end_frame (int): ending frame
    '''
    dataset_type = cfg.dataset.dataset
    dataset_folder = cfg.dataset.data_dir
    resize_img_scale = cfg.dataset.resize_img_scale
    use_aug = cfg.dataset.use_aug
    normalized_scale = cfg.dataset.normalized_scale
    backgroung_color= cfg.dataset.backgroung_color
    patch_size = cfg.dataset.patch_size
    N_patches = cfg.dataset.N_patches
    sample_subject_ratio = cfg.dataset.sample_subject_ratio
    ray_shoot_mode = cfg.dataset.ray_shoot_mode
    box_margin = cfg.dataset.box_margin
    N_samples = cfg.model.neus_renderer.n_samples
    inner_sampling = cfg.dataset.inner_sampling
    res_level = cfg.dataset.res_level
    use_dilated = cfg.dataset.use_dilated
    use_inter_mask = cfg.dataset.use_inter_mask
    
    splits = {
        'train': cfg.dataset.train_split,
        'val': cfg.dataset.val_split,
        'test': cfg.dataset.test_split,
    }
    split = splits[mode]

    if view_split is None:
        view_splits = {
            'train': [str(v) for v in cfg.dataset.train_views],
            'val': [str(v) for v in cfg.dataset.val_views],
            'test': [str(v) for v in cfg.dataset.test_views],
        }
        view_split = view_splits[mode]

    if subsampling_rate is None:
        subsampling_rates = {
            'train': cfg.dataset.train_subsampling_rate,
            'val': cfg.dataset.val_subsampling_rate,
            'test': cfg.dataset.test_subsampling_rate,
        }
        subsampling_rate = subsampling_rates[mode]

    if start_frame is None:
        start_frames = {
            'train': cfg.dataset.train_start_frame,
            'val': cfg.dataset.val_start_frame,
            'test': cfg.dataset.test_start_frame,
        }
        start_frame = start_frames[mode]

    if end_frame is None:
        end_frames = {
            'train': cfg.dataset.train_end_frame,
            'val': cfg.dataset.val_end_frame,
            'test': cfg.dataset.test_end_frame,
        }
        end_frame = end_frames[mode]
    
    if dataset_type == 'zju_mocap':
        dataset = ZJUMoCapDataset_MVS(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            resize_img_scale=resize_img_scale,
            start_frame=start_frame,
            end_frame=end_frame,
            sampling_rate=subsampling_rate,
            views=view_split,
            box_margin=box_margin,
            ray_shoot_mode=ray_shoot_mode,
            backgroung_color=backgroung_color,
            patch_size=patch_size,
            N_patches=N_patches,
            sample_subject_ratio=sample_subject_ratio,
            N_samples=N_samples,
            inner_sampling=inner_sampling,
            res_level=res_level,
            use_dilated=use_dilated,
            use_inter_mask=use_inter_mask,
        )
    elif dataset_type == 'zju_mocap_odp':
        novel_pose_folder = cfg.dataset.novel_pose_folder
        dataset = ZJUMoCapDataset_ODP(
            dataset_folder=dataset_folder,
            subjects=split,
            novel_pose_folder=novel_pose_folder,
            mode=mode,
            resize_img_scale=resize_img_scale,
            start_frame=start_frame,
            end_frame=end_frame,
            sampling_rate=subsampling_rate,
            views=view_split,
            box_margin=box_margin,
            ray_shoot_mode=ray_shoot_mode,
            backgroung_color=backgroung_color,
            N_samples=N_samples,
            inner_sampling=inner_sampling,
            res_level=res_level,
            use_dilated=use_dilated,
            use_inter_mask=use_inter_mask,
        )
    elif dataset_type == 'aistplusplus_odp':
        novel_pose_folder = cfg.dataset.novel_pose_folder
        dataset = AISTPlusPlusDataset_ODP(
            dataset_folder=dataset_folder,
            subjects=split,
            novel_pose_folder=novel_pose_folder,
            mode=mode,
            resize_img_scale=resize_img_scale,
            start_frame=start_frame,
            end_frame=end_frame,
            sampling_rate=subsampling_rate,
            views=view_split,
            box_margin=box_margin,
            ray_shoot_mode=ray_shoot_mode,
            backgroung_color=backgroung_color,
            N_samples=N_samples,
            inner_sampling=inner_sampling,
            res_level=res_level,
            use_dilated=use_dilated,
            use_inter_mask=use_inter_mask,
        )
    elif dataset_type == 'People_Snapshot':
        dataset = PeopleSnapshotDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            resize_img_scale=resize_img_scale,
            start_frame=start_frame,
            end_frame=end_frame,
            sampling_rate=subsampling_rate,
            views=view_split,
            box_margin=box_margin,
            ray_shoot_mode=ray_shoot_mode,
            backgroung_color=backgroung_color,
            patch_size=patch_size,
            N_patches=N_patches,
            sample_subject_ratio=sample_subject_ratio,
            N_samples=N_samples,
            inner_sampling=inner_sampling,
            res_level=res_level,
            use_dilated=use_dilated,
            use_inter_mask=use_inter_mask,
        )
    elif dataset_type == 'SelfreconSynthesis':
        dataset = SelfreconSynthesisDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            resize_img_scale=resize_img_scale,
            start_frame=start_frame,
            end_frame=end_frame,
            sampling_rate=subsampling_rate,
            views=view_split,
            box_margin=box_margin,
            ray_shoot_mode=ray_shoot_mode,
            backgroung_color=backgroung_color,
            patch_size=patch_size,
            N_patches=N_patches,
            sample_subject_ratio=sample_subject_ratio,
            N_samples=N_samples,
            inner_sampling=inner_sampling,
            res_level=res_level,
            use_dilated=use_dilated,
            use_inter_mask=use_inter_mask,
        )
    elif dataset_type == 'SyntheticHuman':
        dataset = SyntheticHumanDataset(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            resize_img_scale=resize_img_scale,
            start_frame=start_frame,
            end_frame=end_frame,
            sampling_rate=subsampling_rate,
            views=view_split,
            box_margin=box_margin,
            ray_shoot_mode=ray_shoot_mode,
            backgroung_color=backgroung_color,
            patch_size=patch_size,
            N_patches=N_patches,
            sample_subject_ratio=sample_subject_ratio,
            N_samples=N_samples,
            inner_sampling=inner_sampling,
            res_level=res_level,
            use_dilated=use_dilated,
            use_inter_mask=use_inter_mask,
        )
    elif dataset_type == 'monocap':
        dataset = MONOCAP(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            resize_img_scale=resize_img_scale,
            start_frame=start_frame,
            end_frame=end_frame,
            sampling_rate=subsampling_rate,
            views=view_split,
            box_margin=box_margin,
            ray_shoot_mode=ray_shoot_mode,
            backgroung_color=backgroung_color,
            patch_size=patch_size,
            N_patches=N_patches,
            sample_subject_ratio=sample_subject_ratio,
            N_samples=N_samples,
            inner_sampling=inner_sampling,
            res_level=res_level,
            use_dilated=use_dilated,
            use_inter_mask=use_inter_mask,
        )
    elif dataset_type == 'h36m':
        dataset = H36M(
            dataset_folder=dataset_folder,
            subjects=split,
            mode=mode,
            resize_img_scale=resize_img_scale,
            start_frame=start_frame,
            end_frame=end_frame,
            sampling_rate=subsampling_rate,
            views=view_split,
            box_margin=box_margin,
            ray_shoot_mode=ray_shoot_mode,
            backgroung_color=backgroung_color,
            patch_size=patch_size,
            N_patches=N_patches,
            sample_subject_ratio=sample_subject_ratio,
            N_samples=N_samples,
            inner_sampling=inner_sampling,
            res_level=res_level,
            use_dilated=use_dilated,
            use_inter_mask=use_inter_mask,
        )
    return dataset