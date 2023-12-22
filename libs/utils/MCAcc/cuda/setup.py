from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='interplate',
      ext_modules=[CUDAExtension('interp2x_boundary2d', ['interp2x_boundary2d.cpp','interp2x_boundary2d_kernel.cu']),
                CUDAExtension('interp2x_boundary3d', ['interp2x_boundary3d.cpp','interp2x_boundary3d_kernel.cu']),
                CUDAExtension('GridSamplerMine', ['GridSamplerMine.cpp','GridSamplerMineKernel.cu']),],
      cmdclass={'build_ext': BuildExtension})

# setup(name='interplate',
#       ext_modules=[CUDAExtension('GridSamplerMine', ['GridSamplerMine.cpp','GridSamplerMineKernel.cu'])],
#       cmdclass={'build_ext': BuildExtension})


# try:
#     from setuptools import setup
# except ImportError:
#     from distutils.core import setup
# from distutils.extension import Extension
# from Cython.Build import cythonize
# from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
# import numpy


# # Get the numpy include directory.
# numpy_include_dir = numpy.get_include()

# # Extensions
# # interp2x_boundary2d
# interp2x_boundary2d = CUDAExtension(
#     'interp2x_boundary2d',
#     sources=[
#         'interp2x_boundary2d.cpp',
#         'interp2x_boundary2d_kernel.cu'
#     ],
#     libraries=['m'],  # Unix-like specific
#     include_dirs=[numpy_include_dir]
# )
# # tinterp2x_boundary3d
# interp2x_boundary3d = CUDAExtension(
#     'interp2x_boundary3d',
#     sources=[
#         'interp2x_boundary3d.cpp',
#         'interp2x_boundary3d_kernel.cu'
#     ],
#     libraries=['m'],  # Unix-like specific
#     include_dirs=[numpy_include_dir]
# )
# # GridSamplerMine
# GridSamplerMine = CUDAExtension(
#     'GridSamplerMine',
#     sources=[
#         'GridSamplerMine.cpp',
#         'GridSamplerMineKernel.cu'
#     ],
#     libraries=['m'],  # Unix-like specific
#     include_dirs=[numpy_include_dir]
# )

# # Gather all extension modules
# ext_modules = [
#     interp2x_boundary2d,
#     interp2x_boundary3d,
#     GridSamplerMine
# ]

# setup(
#     ext_modules=cythonize(ext_modules),
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )
