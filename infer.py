import os
import glob
import torch
import wandb
import logging
import numpy as np
import pytorch_lightning as pl
import argparse
from pyhocon import ConfigFactory

from torch.utils.data import DataLoader
from libs import module_config

# Arguments
parser = argparse.ArgumentParser(
    description='Inference function on with-distribution poses (ZJU training and testing).'
)
parser.add_argument('--conf', type=str, help='Path to config file.')
parser.add_argument('--base_exp_dir', type=str, default=None)
parser.add_argument('--infer_mode', type=str, default='nvs', help='Inference mode, one of the following: [nvs, unseen, odp]. `nvs` for novel \
                    view synthesis on training poses, `unseen` for generalation to unseen poses on novel view, `odp` for generalation to \
                    out-of-distribution poses' )
parser.add_argument('--novel_pose', type=str, default='data/data_prepared/CoreView_392', help='Test specified novel pose, e.g. data/data_prepared/CoreView_392')
parser.add_argument('--novel_pose_view', type=int, default=0, help='Which view to use for novel pose, e.g. 0 for the first view.')
parser.add_argument('--novel_pose_type', type=str, default='zju_mocap_odp', help='The type of novel pose, e.g. zju_mocap_odp, aistplusplus_odp, amass_odp, etc.')
parser.add_argument('--resolution_level', type=int, default=4, help='Test rendering resolution level. e.g. 4(256, 256), 2(512, 512)')
parser.add_argument('--gpus', type=list, default=[0], help='Test on multiple GPUs.')
parser.add_argument('--num-workers', type=int, default=4,
                    help='Number of workers to use for val/test loaders.')
parser.add_argument('--run-name', type=str, default='',
                    help='Run name for Wandb logging.')

if  __name__ == '__main__':
    args = parser.parse_args()
    conf = ConfigFactory.parse_file(args.conf)
    num_workers = args.num_workers
    split_mode = 'val' if args.infer_mode in ['nvs', 'unseen'] else 'test'
    conf['dataset']['res_level'] = args.resolution_level
    
    # validation for novel views synthesis on training poses
    if args.infer_mode == 'nvs':
        conf['dataset'][f'{split_mode}_views'] = [2]
        conf['dataset'][f'{split_mode}_subsampling_rate'] = 2
        conf['dataset'][f'{split_mode}_start_frame'] = 0
        conf['dataset'][f'{split_mode}_end_frame'] = -1
    
    # validation for generalation to unseen poses (teset poses) on novel view
    elif args.infer_mode == 'unseen':
        # conf['dataset'][f'{split_mode}_views'] = [0]
        conf['dataset'][f'{split_mode}_subsampling_rate'] = 50
        conf['dataset'][f'{split_mode}_start_frame'] = 300
        conf['dataset'][f'{split_mode}_end_frame'] = -1
    
    # validation for generalation to out-of-distribution poses
    elif args.infer_mode == 'odp':
        conf['dataset']['dataset'] = args.novel_pose_type
        conf['dataset']['novel_pose_folder'] = args.novel_pose
        conf['dataset'][f'{split_mode}_views'] = [args.novel_pose_view]
        conf['dataset'][f'{split_mode}_subsampling_rate'] = 1
        conf['dataset'][f'{split_mode}_start_frame'] = 0
        conf['dataset'][f'{split_mode}_end_frame'] = -1
    
    dloader = DataLoader(
        module_config.get_dataset(split_mode, conf),
        batch_size=conf.train.batch_size,
        num_workers=conf.train.num_workers,
        shuffle=False)

    # Model
    model = module_config.get_model(conf)

    out_dir = args.base_exp_dir
    # Create logger
    latest_wandb_path = glob.glob(os.path.join(out_dir, 'wandb', 'latest-run', 'run-*.wandb'))
    if len(latest_wandb_path) == 1:
        run_id = os.path.basename(latest_wandb_path[0]).split('.')[0][4:]
    else:
        run_id = None

    if len(args.run_name) > 0:
        run_name = args.run_name
    else:
        run_name = None

    kwargs = {'settings': wandb.Settings(start_method='fork')}
    logger = pl.loggers.WandbLogger(name=run_name,
                                    project='hf-avatar',
                                    id=run_id,
                                    save_dir=out_dir,
                                    config=conf.__dict__,
                                    **kwargs)

    # Create PyTorch Lightning trainer
    checkpoint_path = sorted(glob.glob(os.path.join(out_dir, "checkpoints/epoch*.ckpt")))[-1]
    # checkpoint_path = os.path.join(out_dir, "checkpoints/last.ckpt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError('No checkpoint is found!')

    trainer = pl.Trainer(logger=logger,
                        default_root_dir=out_dir,
                        accelerator='gpu',
                        strategy='ddp' if len(args.gpus) > 1 else None,
                        devices=args.gpus,
                        num_sanity_val_steps=0)
    
    if split_mode == 'test':
        trainer.test(model=model, dataloaders=dloader, ckpt_path=checkpoint_path, verbose=True)
    elif split_mode == 'val':
        trainer.validate(model=model, dataloaders=dloader, ckpt_path=checkpoint_path, verbose=True)