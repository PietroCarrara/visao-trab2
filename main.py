import cv2
import numpy as np

##############
#    DEFS    #
##############

def progress(x, y, w, h):
  if x == 0:
    prog = (y*w + x) / (w*h)
    print(f"\rProgress: {prog * 100:.2}%", end='')

def get_pixel(im, x, y):
  if x >= 0 and y >= 0 and x <= im.shape[0] and y <= im.shape[0]:
    return im[x, y]
  return [0, 0, 0]

def distance_weak(p1, p2):
  assert(len(p1) == len(p2))

  n = len(p1)
  res = 0
  for i in range(n):
    c = float(p1[i]) - float(p2[i])
    res += c*c

  return res

class Match:
  def __init__(self, distance, disparity) -> None:
    self.distance = distance
    self.disparity = disparity

##############
#    MAIN    #
##############

distance = distance_weak

im_left = cv2.imread('cones_l.png')
im_right = cv2.imread('cones_r.png')

assert(im_left.shape == im_right.shape)

WIDTH = im_left.shape[0]
HEIGHT = im_left.shape[1]

# Dimension of square window
WINDOW_SIZE = 3

matches = np.empty([WIDTH, HEIGHT], dtype=Match)

# Loop through the left image
for y in range(HEIGHT):
  for x in range(WIDTH):
    progress(x, y, WIDTH, HEIGHT)

    matches[x, y] = None

    # Run through the right image
    for xr in range(WIDTH):

      # Find the match that minimizes the distance
      d = distance(get_pixel(im_left, x, y), get_pixel(im_right, xr, y))
      if matches[x, y] is None or matches[x, y].distance > d:
        matches[x, y] = Match(d, xr - x)

disparity_map = np.vectorize(lambda m: m.disparity)(matches)