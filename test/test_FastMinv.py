import torch
from FastMinv3x3 import Fast3x3Minv, Fast3x3Minv_backward
from FastMinv4x4 import Fast4x4Minv, Fast4x4Minv_backward
import cv2
import numpy as np
from time import time

from torch.autograd import Function

class FastDiff3x3MinvFunction(Function):
	@staticmethod
	def forward(ctx,input):		
		invs,check=Fast3x3Minv(input)
		ctx.save_for_backward(invs,check)
		ctx.mark_non_differentiable(check)
		return invs,check
	@staticmethod
	def backward(ctx, grad_input, grad_check):
		invs, check = ctx.saved_tensors
		return Fast3x3Minv_backward(grad_input.contiguous(),invs),None

class FastDiff4x4MinvFunction(Function):
	@staticmethod
	def forward(ctx,input):		
		invs,check=Fast4x4Minv(input)
		ctx.save_for_backward(invs,check)
		ctx.mark_non_differentiable(check)
		return invs,check
	@staticmethod
	def backward(ctx, grad_input, grad_check):
		invs, check = ctx.saved_tensors
		return Fast4x4Minv_backward(grad_input.contiguous(),invs),None

def compute_gradient(y, x, grad_outputs=None, retain_graph=True, create_graph=True):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y, requires_grad=False, device=y.device)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, retain_graph=retain_graph, create_graph=create_graph)[0]
    return grad

N=10000
dim=4
device=torch.device("cuda")
ms=torch.randn(N,dim,dim).to(device)
ms.mean() # compute to initialize cuda


# DEFAULT OPS
start=time()
ms_inv = torch.inverse(ms)
end=time()
print("time:", end-start)

errors=(ms_inv.matmul(ms)-torch.eye(dim).to(device).view(1,dim,dim)).norm(dim=(1,2))
print('max:%f, mean:%f.'%(errors.max().item(),errors.mean().item()))

compute_gradient(ms_inv, ms)


# CUSTOM OPS
# torch.cuda.synchronize()
start=time()
if dim==3:
    invs,checks=Fast3x3Minv(ms)
elif dim==4:
    invs,checks=Fast4x4Minv(ms)
# torch.cuda.synchronize()
end=time()
print('%d ms, %d invertible, time:%f'%(checks.numel(),checks.sum().item(),end-start))

errors=(invs.matmul(ms)-torch.eye(dim).to(device).view(1,dim,dim)).norm(dim=(1,2))
print('max:%f, mean:%f.'%(errors.max().item(),errors.mean().item()))

compute_gradient(invs, ms)