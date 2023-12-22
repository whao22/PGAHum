from FastMinv3x3 import Fast3x3Minv, Fast3x3Minv_backward
from FastMinv4x4 import Fast4x4Minv, Fast4x4Minv_backward

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
