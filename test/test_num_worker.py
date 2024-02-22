from time import time
import multiprocessing as mp
import argparse
import logging
from pyhocon import ConfigFactory
from torch.utils.data import DataLoader
import sys
sys.path.append('.')
from libs import module_config


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

def test_num_workers():
    print(f"num of CPU: {mp.cpu_count()}")
    dataset = module_config.get_dataset('train', conf)
    
    for num_workers in range(2, mp.cpu_count(), 2):
        train_dloader = DataLoader(
            dataset,
            batch_size=conf.train.batch_size,
            num_workers=num_workers,
            shuffle=True)
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_dloader):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


if __name__ == '__main__':
    # Args and Conf
    args = parse_arguments()
    conf = ConfigFactory.parse_file(args.conf)
    
    test_num_workers()
