import argparse
import torch
from glob import glob
import wandb
import logging
import os
from pyhocon import ConfigFactory
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from libs import module_config
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--base_exp_dir', type=str, default=None)
    parser.add_argument('--exit_after', type=int, default=-1, help='Checkpoint and exit after specified number of seconds with exit code 2.')
    parser.add_argument('--epochs_per_run', type=int, default=-1, help='Number of epochs to train before restart.')
    parser.add_argument('--run_name', type=str, default='', help='Run name for Wandb logging.')
    args = parser.parse_args()
    
    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    return args


if __name__ == '__main__':
    # Args and Conf
    args = parse_arguments()
    conf = ConfigFactory.parse_file(args.conf)
    
    # Dataset
    train_dloader = DataLoader(
        module_config.get_dataset('train', conf),
        batch_size=conf.train.batch_size,
        num_workers=conf.train.num_workers,
        shuffle=True)

    val_dloader = DataLoader(
        module_config.get_dataset('val', conf),
        batch_size=conf.train.batch_size,
        num_workers=conf.train.num_workers,
        shuffle=False)
    
    # Model
    model = module_config.get_model(conf, args.base_exp_dir)
    checkpoint_callback = ModelCheckpoint(save_top_k=0,
                                          dirpath=os.path.join(conf.general.base_exp_dir, 'checkpoints'),
                                          every_n_epochs=conf.train.save_every_epoch,
                                          save_on_train_epoch_end=True,
                                          save_last=True)
    
    # Logger
    latest_wandb_path = glob(os.path.join(conf.general.base_exp_dir, 'wandb', 'latest-run', 'run-*.wandb'))
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
                                    save_dir=conf.general.base_exp_dir,
                                    config=conf,
                                    **kwargs)
    
    # Trainer
    checkpoint_path = os.path.join(conf.general.base_exp_dir, 'checkpoints/last.ckpt')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = None

    epochs_per_run = args.epochs_per_run
    max_epochs = conf.train.max_epochs
    if epochs_per_run <= 0:
        # epochs_per_run is not specified: we train with max_epochs and validate
        # this usually applies for training on local machines
        pass
    else:
        # epochs_per_run is specified: we train with already trained epochs + epochs_per_run,
        # and do not validate
        # this usually applies for training on HPC cluster with jon-chaining
        if checkpoint_path is None:
            max_epochs = epochs_per_run
        else:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            max_epochs = min(ckpt['epoch'] + epochs_per_run, max_epochs)
            del ckpt

        validate_every = max_epochs + 1
        
    trainer = pl.Trainer(logger=logger,
                        log_every_n_steps=conf.train.log_every_step,
                        default_root_dir=conf.general.base_exp_dir,
                        callbacks=[checkpoint_callback],
                        max_epochs=max_epochs,
                        check_val_every_n_epoch=conf.train.val_every_epoch,
                        accelerator='gpu',
                        strategy='ddp' if len(conf.train.gpus) > 1 else None,
                        devices=conf.train.gpus,
                        num_sanity_val_steps=0)
    
    trainer.fit(model=model, train_dataloaders=train_dloader, val_dataloaders=val_dloader, ckpt_path=checkpoint_path)
