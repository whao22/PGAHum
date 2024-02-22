from torch.utils.cpp_extension import load
import torch
from pkg_resources import parse_version

# import sys
# sys.path.append("")
import custom_ops
import os

gridsample_grad2 = None


def _init():
    global gridsample_grad2
    if gridsample_grad2 is None:
        gridsample_grad2 = custom_ops.get_plugin(
            module_name='gridsample_grad',
            sources=['cuda/grid_sample.cpp', 'cuda/grid_sample.cu'],
            headers=None,
            source_dir=os.path.dirname(__file__),
            extra_cuda_cflags=['--use_fast_math'],
        )
    return True

_init()