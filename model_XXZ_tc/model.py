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

Jz = 1
Jx = 2

h = np.ones(num_sites)

TX = np.zeros((dim,dim),dtype = complex)
TY = np.zeros((dim,dim),dtype = complex)
TZ = np.zeros((dim,dim),dtype = complex)

for i in range(num_sites):
    TZ = TZ + h[i]*mycode.SingleZ(i,num_sites)


for i in range(num_sites-1):
    TX = TX + Jx*mycode.XX_pair(num_sites,i,i+1)
    TY = TY + Jx*mycode.YY_pair(num_sites,i,i+1)
    TZ = TZ + Jz*mycode.ZZ_pair(num_sites,i,i+1)

H_XXZ = TX + TY + TZ

Sum_Z = np.zeros((dim,dim),dtype = complex)
for j in range(num_sites):
    Sum_Z = Sum_Z + mycode.SingleZ(j,num_sites)
Parity = scipy.linalg.expm(1j*math.pi*Sum_Z/2)

norm = mycode.matrix_norm(H_XXZ)
ham = math.pi * H_XXZ/norm
TX = math.pi * TX/norm 
TY = math.pi * TY/norm 
TZ = math.pi * TZ/norm

np.save('TX.npy',TX)
np.save('TY.npy',TY)
np.save('TZ.npy',TZ)





eig,vec = np.linalg.eig(ham)
ham0 = (ham + ham.dot(Parity))/2
ham1 = (ham - ham.dot(Parity))/2




beta = 1
thermal_state = scipy.linalg.expm(-beta * ham)
thermal_state_sym = scipy.linalg.expm(-beta * ham0)
thermal_state_asym = scipy.linalg.expm(-beta * ham1)

A = mycode.SingleX(0,num_sites)


np.save('ham.npy',ham)
np.save('ham_sym.npy',0.5*(ham + ham.dot(Parity)))
np.save('ham_asym.npy',0.5*(ham - ham.dot(Parity)))
np.save('thermal_state.npy',thermal_state)
np.save('thermal_state_sym.npy',thermal_state_sym)
np.save('thermal_state_asym.npy',thermal_state_asym)
np.save('A.npy',A)
np.save('Parity.npy',Parity)

d = len(ham[0])
Z = mycode.trace(thermal_state)
ZS = mycode.trace(thermal_state_sym) - 0.5*d
ZA = mycode.trace(thermal_state_asym) - 0.5*d

print(abs(Z - ZS - ZA))

parity_sym_sum = mycode.trace(thermal_state_sym.dot(Parity))
parity_asym_sum = mycode.trace(thermal_state_asym.dot(Parity))

NA = ZS - parity_sym_sum 
NS = ZA + parity_asym_sum 

Parameters = np.array([ZS,ZA,NS,NA])
print(Parameters)
np.save('Parameters.npy',Parameters)