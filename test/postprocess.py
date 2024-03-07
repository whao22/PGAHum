from PIL import Image

def fill_zeros_with_nearest_pixels(image):
    width, height = image.size
    pixels = image.load()

    for x in range(width):
        for y in range(height):
            if pixels[x, y] == (0, 0, 0):
                # Get nearest non-zero pixel
                nearest_pixel = find_nearest_nonzero_pixel(image, x, y)
                pixels[x, y] = nearest_pixel

    return image

def find_nearest_nonzero_pixel(image, x, y):
    width, height = image.size
    pixels = image.load()

    for d in range(1, max(width, height)):
        for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                if pixels[nx, ny] != (0, 0, 0):
                    return pixels[nx, ny]

    # If no non-zero pixels are found, return black
    return (0, 0, 0)




# Load the image
input_image = Image.open("exp/CoreView_377_1709393779_slurm_mvs_1_1_3_true/wandb/run-20240304_221523-wenlnewl/files/media/images/validation_samples_221663_85667b89a7b0e025d97f.png")

# Fill zeros with nearest pixels
output_image = fill_zeros_with_nearest_pixels(input_image)

# Save the filled image
output_image.save("output_image.jpg")
