# import torch

# a = torch.rand(1, 65536, 64)

# a_list = torch.split(a, 4096, dim=1)

# for aa in a_list:
#     print(aa.shape)

import torch

# 定义一个3x3的矩阵
kernel_matrix = torch.tensor([[1, 0, -1],
                              [1, 0, -1],
                              [1, 0, -1]], dtype=torch.float32)

# 随机生成一个256x256x3的图像
input_image = torch.rand(1, 3, 256, 256)

# 将矩阵扩展为与图像的通道数相匹配
expanded_kernel = kernel_matrix.unsqueeze(0).unsqueeze(0).expand(3, 3, 3, 3)

# 使用torch.matmul函数进行矩阵相乘
output = torch.matmul(input_image, expanded_kernel)

print(output.shape)  # 输出相乘后的结果形状
