import cv2

image_path = "data/data_prepared/male_outfit2/0/images/frame_000129.png"

image = cv2.imread(image_path)

new_image = image[::4,::4]

cv2.imwrite("aaa.png", new_image)
print()