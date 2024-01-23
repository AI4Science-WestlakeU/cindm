import re
import numpy as np
from matplotlib import pyplot as plt
import os

for sim in range(5):
  # path="/home/ubuntu/lily-pad/LilyPad/saved/naca_ellipse_test_{0}.txt".format(sim)
  # path_vel="/home/ubuntu/lily-pad/LilyPad/dataset/naca_ellipse/test_trajectories/"
  # path_prs="/home/ubuntu/lily-pad/LilyPad/dataset/naca_ellipse/test_trajectories/"
  path="./naca_ellipse_train_{0}.txt".format(sim)
  path_vel="../dataset/naca_ellipse_multiple/training_trajectories/"
  path_prs="../dataset/naca_ellipse_multiple/training_trajectories/"

  try:
    os.makedirs(path_vel+"sim_{:06d}".format(sim))
  except Exception:
    pass

  f = open(path)
  x_coords = []
  y_coords = []
  pressure = []
  jp = 0
  j = 0
  m = 62
  for s_line in f:
      l = list(re.split(' | ;|\n', s_line))
      if l[0] == "x-coords":
          col = []
          for i in range(1, len(l)-2):
              col.append(float(l[i]))
          x_coords.append(col)
      elif l[0] == "y-coords":
          col = []
          for i in range(1, len(l)-2):
              col.append(float(l[i]))
          y_coords.append(col)
      elif l[0] == "pressure":
          jp += 1
          col = []
          for i in range(1, len(l)-2):
              col.append(float(l[i]))
          pressure.append(col)
      if jp == m:
          velocity = np.stack((np.array(x_coords), np.array(y_coords)))
          velocity.shape
          np.save(path_vel+"sim_{:06d}/velocity_{:06d}".format(sim, j), velocity)
          np_pressure = np.array(pressure)
          np.save(path_prs+"sim_{:06d}/pressure_{:06d}".format(sim, j), np_pressure)
          jp = 0
          j += 1
          x_coords = []
          y_coords = []
          pressure = []
