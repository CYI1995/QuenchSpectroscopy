import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random

import scipy.linalg
import source as mycode
import statistics
import cmath

# def h(eps,k_temp,t):

#     return eps*math.cos(k_temp*t)

def h(t):

    if(t < 10*math.pi):

        return math.exp(-2*t*t/(math.pi*math.pi))*math.cos(math.pi*t)

    else: 
        return 0

def JordanWigner_ci(i,num):

    dim = 2**num

    X = mycode.SingleX(i,num)
    Z = mycode.SingleZ(i,num)
    Y = 1j*X.dot(Z)

    Sum_Z = np.zeros((dim,dim),dtype = complex)
    for j in range(i):
        Sum_Z = Sum_Z + mycode.SingleZ(j,num) - np.identity(dim)
    Product_Z = scipy.linalg.expm(1j*math.pi*Sum_Z/2)


    return Product_Z.dot(X + 1j*Y)/2

def JordanWinger_neighbor_hopping(i,num):


    XX = mycode.XX_pair(num,i,i+2)
    YY = mycode.YY_pair(num,i,i+2)
    Z = mycode.SingleZ(i+1,num)

    return 0.5*(XX + YY).dot(Z)

def JordanWigner_ni(i,num):

    dim = 2**num 

    return 0.5*(np.identity(dim) - mycode.SingleZ(i,num))


def find_closest_idx(energy_data,k):

    new_list = np.zeros(len(energy_data))
    for i in range(len(energy_data)):
        new_list[i] = abs(energy_data[i] - k)

    return np.argmin(new_list)


def signal(frequencies, amplitudes,t):

    s = 0 + 1j*0 
    L = len(frequencies)

    for l in range(L):
        s = s + amplitudes[l] * cmath.exp(1j*frequencies[l] * t)

    return s



ham = np.load('ham.npy')
A = np.load('A0.npy')
P = np.load('Parity.npy')
thermal_state = np.load('thermal_state.npy')


dim = len(ham[0])

Parameters = np.load('Parameters.npy')
ZS = Parameters[0]
ZA = Parameters[1]
NS = Parameters[2]
NA = Parameters[3]
Z = ZS + ZA



Amplitudes = [] 
Energy_gaps = []

eig,vec = np.linalg.eig(ham)

idx_gs = np.argsort(eig)[0]
vec_gs = vec[:,idx_gs]
E0 = eig[idx_gs].real
for m in range(dim):

    print(m)

    Em = eig[m].real 
    vec_m = vec[:,m]

    amplitude = abs(np.vdot(vec_m,A.dot(vec_gs)))**2
    energy_gap = E0 - Em

    if(abs(amplitude) > 0.01):
        Amplitudes.append(amplitude)
        Energy_gaps.append(energy_gap)

np.save('Amplitudes_GS.npy',Amplitudes)
np.save('Energy_gap_GS.npy',Energy_gaps)






