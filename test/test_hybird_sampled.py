import numpy as np

x = np.arange(0, 128)
y = np.arange(0, 1024)
xv, _ = np.meshgrid(x, y)
print(xv)

n = np.tile((np.random.rand(1024, 1)*128).astype(np.int32), [1, 128])
print(n)

print(xv < n)