import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random

import scipy.linalg
import source as mycode
import statistics


num_sites = 8
dim = 2**num_sites

W = 10
Jz = 1
Jx = 2

h = np.ones(num_sites)

TX = np.zeros((dim,dim),dtype = complex)
TY = np.zeros((dim,dim),dtype = complex)
TZ = np.zeros((dim,dim),dtype = complex)

for i in range(num_sites):
    TZ = TZ + h[i]*mycode.SingleZ(i,num_sites)


for i in range(num_sites-1):
    TX = TX - Jx*mycode.XX_pair(num_sites,i,i+1)
    TY = TY - Jx*mycode.YY_pair(num_sites,i,i+1)
    TZ = TZ - Jz*mycode.ZZ_pair(num_sites,i,i+1)

norm = mycode.matrix_norm(TX + TY + TZ)

TX = math.pi * TX/norm 
TY = math.pi * TY/norm 
TZ = math.pi * TZ/norm

np.save('TX.npy',TX)
np.save('TY.npy',TY)
np.save('TZ.npy',TZ)

