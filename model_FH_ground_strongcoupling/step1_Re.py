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

def Generate_Gaussian_noise(dim):

    Gaussian_noise_real = np.random.randn(dim,dim)
    Gaussian_noise_imag = np.random.randn(dim,dim)
    for i in range(dim):
        Gaussian_noise_real[i][i] = 0
        Gaussian_noise_imag[i][i] = 0
    Gaussian_noise = Gaussian_noise_real + 1j*Gaussian_noise_imag 
    Gaussian_noise = 0.5*(Gaussian_noise + np.conj(Gaussian_noise).T)/(dim**(1.5))

    return Gaussian_noise


ham = np.load('ham.npy')
ham_sym = np.load('ham_sym.npy')
ham_asym = np.load('ham_asym.npy')
A = np.load('A0.npy')
P = np.load('Parity.npy')
thermal_state = np.load('thermal_state.npy')
thermal_state_sym = np.load('thermal_state_sym.npy')
thermal_state_asym = np.load('thermal_state_asym.npy')

T0 = np.load('T0.npy')
T1 = np.load('T1.npy')
T2 = np.load('T2.npy')

dim = len(ham[0])

Parameters = np.load('Parameters.npy')
ZS = Parameters[0]
ZA = Parameters[1]
NS = Parameters[2]
NA = Parameters[3]
Z = ZS + ZA

num_T = 1000
nT = 50
T= nT*math.pi
dt = T/num_T


T_axis = np.zeros(num_T)
O0_axis = np.zeros(num_T)
O1_axis = np.zeros(num_T)
O2_axis = np.zeros(num_T) 
O3_axis = np.zeros(num_T)
O4_axis = np.zeros(num_T)
O5_axis = np.zeros(num_T) 
O6_axis = np.zeros(num_T)

init_state = thermal_state/Z
init_state_sym = thermal_state_sym/(ZS + NA)
init_state_asym = thermal_state_asym/(ZA + NS)

noisy_init_state_sym = np.load('new_init_state_sym.npy')
noisy_init_state_asym = np.load('new_init_state_asym.npy')


Kick_Operator2 = (P + A)/math.sqrt(2)
kicked_state_sym = Kick_Operator2.dot((noisy_init_state_sym).dot(np.conj(Kick_Operator2).T))
kicked_state_asym = Kick_Operator2.dot((noisy_init_state_asym).dot(np.conj(Kick_Operator2).T))


U_Tro_dt = scipy.linalg.expm(-1j*T2*dt)
U_Tro_dt = scipy.linalg.expm(-1j*T1*dt).dot(U_Tro_dt)
U_Tro_dt = scipy.linalg.expm(-1j*T0*dt).dot(U_Tro_dt)

U_Ext_dt = scipy.linalg.expm(-1j*ham*dt)

At_Tro = A.copy()
At_Ext = A.copy()

for n in range(num_T):

    print(n)

    t = (n+1)*dt
    At_Tro = (np.conj(U_Tro_dt).T).dot(At_Tro.dot(U_Tro_dt))
    At_Ext = (np.conj(U_Ext_dt).T).dot(At_Ext.dot(U_Ext_dt))
    AAt = A.dot(At_Ext)

    v0 = Z * mycode.trace(init_state.dot(AAt)).real
    v1 = (NA + ZS)* mycode.trace(kicked_state_sym.dot(At_Tro)).real
    v2 = (NS + ZA)* mycode.trace(kicked_state_asym.dot(At_Tro)).real
    v3 = (NA + ZS)* mycode.trace(kicked_state_sym.dot(At_Ext)).real
    v4 = (NS + ZA)* mycode.trace(kicked_state_asym.dot(At_Ext)).real

    T_axis[n] = t
    O0_axis[n] = v0/Z
    O1_axis[n] = v1/Z
    O2_axis[n] = v2/Z
    O3_axis[n] = v3/Z
    O4_axis[n] = v4/Z
    O5_axis[n] = mycode.trace(A.dot(At_Tro)).real/Z


plt.plot(T_axis, O0_axis,label = 'Exact')
plt.plot(T_axis, O1_axis - O2_axis + O5_axis,label = 'Noisy + Trotter')
plt.plot(T_axis, O3_axis - O4_axis + O5_axis,label = 'Noisy + Ext')

np.save('signal_Re_ExtExt.npy',O0_axis)
np.save('signal_Re_NoiTro.npy',O1_axis - O2_axis + O5_axis)
np.save('signal_Re_NoiExt.npy',O3_axis - O4_axis + O5_axis)


plt.legend()
plt.show()

