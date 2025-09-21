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

def trace_distance(r1,r2):

    d = len(r1[0])

    dr = r1 - r2 

    eig,vec = np.linalg.eig(dr)
    output = 0
    for i in range(d):
        output = output + abs(eig[i])

    return output /2

def matrix_one_norm(M):

    eig,vec = np.linalg.eig(M)
    norm = 0 
    for i in range(dim):
        norm = norm + abs(eig[i])

    return norm

ham = np.load('ham.npy')
thermal_state = np.load('thermal_state.npy')
thermal_state_sym = np.load('thermal_state_sym.npy')
thermal_state_asym = np.load('thermal_state_asym.npy')
dim = len(ham[0])

Parameters = np.load('Parameters.npy')
ZS = Parameters[0].real
ZA = Parameters[1].real
NS = Parameters[2].real
NA = Parameters[3].real


# Gaussian_noise_real = np.random.randn(dim,dim)
# Gaussian_noise_imag = np.random.randn(dim,dim)
# for i in range(dim):
#     Gaussian_noise_real[i][i] = 0
#     Gaussian_noise_imag[i][i] = 0
# Gaussian_noise = Gaussian_noise_real + 1j*Gaussian_noise_imag 
# Gaussian_noise = 0.5*(Gaussian_noise + np.conj(Gaussian_noise).T)
# onenorm = matrix_one_norm(Gaussian_noise)
# Gaussian_noise = Gaussian_noise/onenorm 

# np.save('normalized_Gaussian_noise.npy',Gaussian_noise)

Noise_Hamiltonian = np.load('normalized_Gaussian_noise.npy')


eps = 0.1
Gaussian_Noise = scipy.linalg.expm(Noise_Hamiltonian)
trace = mycode.trace(Gaussian_Noise)
Gaussian_Noise = Gaussian_Noise/trace

thermal_state = thermal_state/(ZS + ZA)
noisy_init_state = thermal_state + eps * Gaussian_Noise 
new_init_state = noisy_init_state/mycode.trace(noisy_init_state)
np.save('new_init_state.npy',new_init_state)

thermal_state_sym = thermal_state_sym/(ZS + NA)
noisy_init_state = thermal_state_sym + eps * Gaussian_Noise 
new_init_state_sym = noisy_init_state/mycode.trace(noisy_init_state)
np.save('new_init_state_sym.npy',new_init_state_sym)

thermal_state_asym = thermal_state_asym/(ZA + NS)
noisy_init_state = thermal_state_asym + eps * Gaussian_Noise 
new_init_state_asym = noisy_init_state/mycode.trace(noisy_init_state)
np.save('new_init_state_asym.npy',new_init_state_asym)
