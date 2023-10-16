import jax as jax
import jax.numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr
from tqdm import tqdm

N_ITER=15000
REYNOLDS_NO=80

N_POINTS_X=300
N_POINTS_Y=50

OBJ_CENTER_X=N_POINTS_X//5
OBJ_CENTER_Y=N_POINTS_Y//2
OBJ_RADII_IDX=N_POINTS_Y//9

VMAX_IN_X=0.04
VISUALISE=True
PLOT_EVERY=100
N_FIRST_SKIPS=0

'''LBM Grid: D2Q9
    6   2   5
      \ | /
    3 - 0 - 1
      / | \
    7   4   8 
'''

N_DISCRETE=9

LATTICE_VELS=np.array([
    [0,1,0,-1,0,1,-1,-1,1],
    [0,0,1,0,-1,1,1,-1,-1]
    ])
LATTICE_IDX=np.array([0,1,2,3,4,5,6,7,8])
LATTICE_OPP_IDX=np.array([0,3,4,1,2,7,8,5,6])
LATTICE_WEIGHTS = np.array([
    4/9,                        # Center Velocity [0,]
    1/9,  1/9,  1/9,  1/9,      # Axis-Aligned Velocities [1, 2, 3, 4]
    1/36, 1/36, 1/36, 1/36,     # 45 ° Velocities [5, 6, 7, 8]
])

V_RIGHT=np.array([1,5,8])
V_UP=np.array([2,5,6])
V_LEFT=np.array([3,6,7])
V_DOWN=np.array([4,7,8])

ONLY_X=np.array([0,1,3])
ONLY_Y=np.array([0,2,4])