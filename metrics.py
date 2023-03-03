import cv2
import numpy as np

##############
#   PARAMS   #
##############

img_ground = cv2.imread('cones_l_disp.png').astype(float)
img_out = cv2.imread('out/cones_5_robust.png').astype(float)

# Max error that is forgiven
GRACE = 5

##############
#    MAIN    #
##############

# Make sure both images have the same dimensions
assert(img_ground.shape[:2] == img_out.shape[:2])

# Discard color channels, assume the images are grayscale and use only the red channel
if (len(img_ground.shape) > 2):
  img_ground = img_ground[:, :, 0]
if (len(img_out.shape) > 2):
  img_out = img_out[:, :, 0]

WIDTH = img_ground.shape[0]
HEIGHT = img_ground.shape[1]

img_diff = np.zeros([WIDTH, HEIGHT], dtype=np.uint8)
err_count = 0
for y in range(HEIGHT):
  for x in range(WIDTH):
    err = abs(img_ground[x, y] - img_out[x, y])
    if err > GRACE:
      err_count += 1
      img_diff[x, y] = err - GRACE

# Remap errors to [0, 255] range
max = np.max(img_diff)
for y in range(HEIGHT):
  for x in range(WIDTH):
    img_diff[x, y] = round(img_diff[x, y]*255/max)

print(f"Total bad pixel count: {err_count}")
print(f"Bad pixel percentage: {err_count*100 / (WIDTH*HEIGHT):.2f}%")

print(f"Saving error map as errors.png...")
cv2.imwrite('errors.png', img_diff)
print("Done!")