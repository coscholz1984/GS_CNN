#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

__author__     = "Christian Scholz and Sandy Scholz"
__copyright__  = "Copyright 2021, authors"
__credits__    = ["Christian Scholz", "Sandy Scholz"]
__license__    = "GPLv3"
__version__    = "1.0"
__maintainer__ = "Christian Scholz and Sandy Scholz"
__email__      = "coscholz1984@gmail.com"
__status__     = "Development"
__summary__    = "Interactively view patterns in the Gray-Scott model in 2D"

"""

import matplotlib.pyplot as plt
from scipy import ndimage
import GStools as gst
import numpy as np
import time
import sys

# Draw settings and timer
tdraw = 1 / 30  # draw all tdraw seconds
nextframe = time.time() + tdraw

# Default parameters
seed = 2
diff_u = 0.2
diff_v = 0.1
rate_k = 0.045
rate_f = 0.009
timestep = 0.25
stoptime = 15000
sys_width = 128

# parse input parameters
if len(sys.argv) == 3:
    rate_f = float(sys.argv[1])
    rate_k = float(sys.argv[2])
elif len(sys.argv) == 4:
    diff_u = float(sys.argv[1])
    rate_f = float(sys.argv[2])
    rate_k = float(sys.argv[3])
elif len(sys.argv) == 5:
    seed = int(sys.argv[1])
    diff_u = float(sys.argv[2])
    rate_f = float(sys.argv[3])
    rate_k = float(sys.argv[4])

# setup initial values and initial state
np.random.seed(seed)

rate_sum = rate_k + rate_f

shift_x = 0.5
shift_y = 0.5
while not gst.check(gst.chessboarddistance(shift_x, shift_y, 0.5, 0.5)):
	shift_x = np.random.rand() * 0.8 + 0.1  # 0.3
	shift_y = np.random.rand() * 0.8 + 0.1  # 0.2

# construct initial state made of two perturbations of a homogeneous
# background at random distance within a certain range(see check).
timer, U, V = gst.initPearsonNoise(sys_width, shift_x, shift_y, 0)
# random rotation
angle = np.random.randint(0, 89)
rot_A = ndimage.rotate(U, angle, mode="wrap", prefilter=False, reshape=False)
rot_B = ndimage.rotate(V, angle, mode="wrap", prefilter=False, reshape=False)

# random shift in x and y
x_rand = int(np.random.rand() * sys_width)
y_rand = int(np.random.rand() * sys_width)
U = np.roll(rot_A, x_rand, axis=1)
V = np.roll(rot_B, x_rand, axis=1)
U = np.roll(rot_A, y_rand, axis=0)
V = np.roll(rot_B, y_rand, axis=0)

# setup figure and labels
Iobj = plt.imshow(U, "CMRmap")
plt.annotate(xy=[0, 0], xytext=(0, 5), textcoords="offset points", s="D_u: " + "{:.2f}".format(diff_u))
plt.annotate(xy=[0, 0], xytext=(60, 5), textcoords="offset points", s="f: " + "{:.4f}".format(rate_f))
plt.annotate(xy=[0, 0], xytext=(120, 5), textcoords="offset points", s="k: " + "{:.4f}".format(rate_k))
Lmin = plt.annotate(xy=[0, 0], xytext=(210, 15), textcoords="offset points", s="min: 0.10")
Lmax = plt.annotate(xy=[0, 0], xytext=(210, 5), textcoords="offset points", s="max: 1.00")
plt.clim(0.1, 1.0)
plt.pause(0.000000001)

# Main integration loop
while timer < stoptime:
    nonlinearInteraction = gst.nLI(U, V)
    U_new = (
		U
		+ (
			diff_u * gst.laplace(U)
			- nonlinearInteraction
			+ rate_f * (1 - U)
		)
		* timestep
	)
    V_new = (
		V
		+ (diff_v * gst.laplace(V) + nonlinearInteraction - rate_sum * V)
		* timestep
	)
    U = U_new
    V = V_new
    timer = timer + timestep
    # Update figure and display current U
    if time.time() > nextframe:
        Iobj.set_data(U)
        Lmin.set_text("min: " + "{:.2f}".format(U.min(axis=None)))
        Lmax.set_text("max: " + "{:.2f}".format(U.max(axis=None)))
        plt.pause(0.000000001)
        nextframe = time.time() + tdraw