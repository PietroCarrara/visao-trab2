import cv2
import itertools
import numpy as np
import multiprocessing as mp
from math import sqrt


##############
#    DEFS    #
##############

def get_pixel(im, x, y):
  if x >= 0 and y >= 0 and x < im.shape[0] and y < im.shape[0]:
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

# Main function
def process_step(xy):
  x, y = xy
  match: Match = None

  # Run through the right image
  for xr in range(WIDTH):
    # Run through the window accumulating the distance
    d = 0
    for wy in range(WINDOW_SIZE):
      for wx in range(WINDOW_SIZE):
        d += distance(get_pixel(im_left, x+wx, y+wy), get_pixel(im_right, xr+wx, y+wy))

    # Find the match that minimizes the distance
    if match is None or match.distance > d:
      match = Match(d, xr - x)

  return x, y, match

class Match:
  def __init__(self, distance, disparity) -> None:
    self.distance = distance
    self.disparity = disparity

##############
#    MAIN    #
##############

distance = distance_weak

im_left = cv2.cvtColor(cv2.imread('cones_l.png'), cv2.COLOR_BGR2LAB)
im_right = cv2.cvtColor(cv2.imread('cones_r.png'), cv2.COLOR_BGR2LAB)

assert(im_left.shape == im_right.shape)

WIDTH = im_left.shape[0]
HEIGHT = im_left.shape[1]

# Dimension of square window
WINDOW_SIZE = 5

matches = np.empty([WIDTH, HEIGHT], dtype=Match)
with mp.Pool() as pool:
  # for x in WIDTH, y in HEIGHT
  args = itertools.product(range(WIDTH), range(HEIGHT))

  iters = 0
  total = WIDTH*HEIGHT

  for x, y, match in pool.imap_unordered(process_step, args, 16):
    matches[x, y] = match
    # Inform about progress
    iters += 1
    if iters % 100 == 1:
      print(f"\rProgress: {iters/total * 100:.2f}%", end='')

disparity_map = np.vectorize(lambda m: m.disparity)(matches)
print("\nDone!")

# Remap values to [0, 255] range, so the map can be displayed as an image
min = np.min(disparity_map)
max = np.max(disparity_map)
im_disp = np.zeros([WIDTH, HEIGHT], dtype=np.uint8)

for y in range(HEIGHT):
  for x in range(WIDTH):
    im_disp[x, y] = round((disparity_map[x, y] - min) * 255 / (max-min))
cv2.imwrite('out.png', im_disp)