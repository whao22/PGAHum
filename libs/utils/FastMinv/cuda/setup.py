from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
setup(name='FastMinv',
      ext_modules=[CUDAExtension('FastMinv3x3', ['M3x3Inv.cpp','Matrix3x3InvKernels.cu']),
                   CUDAExtension('FastMinv4x4', ['M4x4Inv.cpp','Matrix4x4InvKernels.cu']),],
      cmdclass={'build_ext': BuildExtension})