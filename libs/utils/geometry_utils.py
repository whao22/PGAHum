'''
borrowed from NeuS.
'''
import mcubes
import torch
import numpy as np
from libs.utils.general_utils import sample_sdf_from_grid

def compute_gradient(y, x, grad_outputs=None, retain_graph=True, create_graph=True):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y, requires_grad=False, device=y.device)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, retain_graph=retain_graph, create_graph=create_graph)[0]
    return grad

def lambda_sdf(points, sdf_network, sdf_kwargs, use_init_sdf=True):
    sdf_nn_output = sdf_network(points)
    sdf = sdf_nn_output[:, :1]
    
    if use_init_sdf:
        init_sdf = sample_sdf_from_grid(points, **sdf_kwargs)
        sdf = init_sdf + sdf
        
    return sdf

def extract_fields(bound_min, bound_max, resolution, sdf_network, sdf_kwargs, device):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).to(device).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).to(device).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).to(device).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    points = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = lambda_sdf(points, sdf_network, sdf_kwargs).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, sdf_network, sdf_kwargs, device):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, sdf_network, sdf_kwargs, device)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles