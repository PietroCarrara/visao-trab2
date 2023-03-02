import cv2
import itertools
import numpy as np
import multiprocessing as mp

##############
#    DEFS    #
##############

def get_pixel(im, x, y):
  if x >= 0 and y >= 0 and x < im.shape[0] and y < im.shape[0]:
    return im[x, y].astype(float)
  return np.zeros(3)

def distance_weak(xy1, xy2):
  d = 0
  x1, y1 = xy1
  x2, y2 = xy2
  for wy in range(WINDOW_SIZE):
      for wx in range(WINDOW_SIZE):
        p1 = get_pixel(im_left, x1+wx, y1+wy)
        p2 = get_pixel(im_right, x2+wx, y2+wy)
        for i in range(len(p1)):
          d += (p1[i] - p2[i])**2

  return d

def distance_robust(xy1, xy2):
  d = 0
  x1, y1 = xy1
  x2, y2 = xy2
  for wy in range(WINDOW_SIZE):
      for wx in range(WINDOW_SIZE):
        p1 = get_pixel(im_left, x1+wx - WINDOW_SIZE//2, y1+wy - WINDOW_SIZE//2)
        p2 = get_pixel(im_right, x2+wx - WINDOW_SIZE//2, y2+wy - WINDOW_SIZE//2)

        err = 0
        for i in range(len(p1)):
          err += abs(p1[i] - p2[i])

        # Função robusta
        e = 30
        errs = err**2
        d += errs / (errs + e**2)

  return d

# Main function
def process_step(xy):
  # Given a position on the left image
  x, y = xy

  # Run through the right image and find the best match
  match: Match = Match(np.Inf, np.Inf)
  for xr in range(WIDTH):
    # Find the match that minimizes the distance
    d = distance((x,y), (xr,y))
    if match.distance > d:
      match.distance = d
      match.disparity = x - xr

  return x, y, match

class Match:
  def __init__(self, distance, disparity) -> None:
    self.distance = distance
    self.disparity = disparity

##############
#   PARAMS   #
##############

distance = distance_robust

# Dimension of the square window
WINDOW_SIZE = 3

im_left = cv2.cvtColor(cv2.imread('cones_l_small.png'), cv2.COLOR_BGR2LAB)
im_right = cv2.cvtColor(cv2.imread('cones_r_small.png'), cv2.COLOR_BGR2LAB)

##############
#    MAIN    #
##############

assert(im_left.shape == im_right.shape)

WIDTH = im_left.shape[0]
HEIGHT = im_left.shape[1]

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
    im_disp[x, y] = 255 - round((disparity_map[x, y] - min) * 255 / (max-min))
cv2.imwrite('out.png', im_disp)