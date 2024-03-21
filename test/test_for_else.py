# N = 2

# for i in range(N):
#     print(i)
# else:
#     print("N is", N)

# print()

import cv2

img = cv2.imread('data/SyntheticHuman/malcolm/mask/04/000059.png')

cv2.imwrite('aaa1.png', img*255)