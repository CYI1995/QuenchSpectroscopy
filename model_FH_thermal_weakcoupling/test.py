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


ham = np.load('ham.npy')
ham_sym = np.load('ham_sym.npy')
ham_asym = np.load('ham_asym.npy')
A = np.load('A0.npy')
P = np.load('Parity.npy')
thermal_state = np.load('thermal_state.npy')
thermal_state_sym = np.load('thermal_state_sym.npy')
thermal_state_asym = np.load('thermal_state_asym.npy')

dim = len(ham[0])

T0 = np.load('T0.npy')
T1 = np.load('T1.npy')
T2 = np.load('T2.npy')
T3 = np.load('T3.npy')


num_T = 1000
nT = 50
T= nT*math.pi
dt = T/num_T

print(mycode.matrix_norm(ham - T0 - T1 - T2 - T3))

# U_Tro_dt = scipy.linalg.expm(-1j*T3*dt)
# U_Tro_dt = scipy.linalg.expm(-1j*T2*dt).dot(U_Tro_dt)
# U_Tro_dt = scipy.linalg.expm(-1j*T1*dt).dot(U_Tro_dt)
# U_Tro_dt = scipy.linalg.expm(-1j*T0*dt).dot(U_Tro_dt)

# U_Ext_dt = scipy.linalg.expm(-1j*ham*dt)

# print(mycode.matrix_norm(U_Tro_dt - U_Ext_dt))



