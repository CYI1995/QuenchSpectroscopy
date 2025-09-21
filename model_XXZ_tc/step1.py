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
A = np.load('A.npy')
P = np.load('Parity.npy')
thermal_state = np.load('thermal_state.npy')
thermal_state_sym = np.load('thermal_state_sym.npy')
thermal_state_asym = np.load('thermal_state_asym.npy')

dim = len(ham[0])

T0 = np.load('TX.npy')
T1 = np.load('TY.npy')
T2 = np.load('TZ.npy')


Parameters = np.load('Parameters.npy')
ZS = Parameters[0].real
ZA = Parameters[1].real
NS = Parameters[2].real
NA = Parameters[3].real
Z = ZS + ZA

num_T = 1000
nT = 50
T= nT*math.pi
dt = T/num_T


T_axis = np.zeros(num_T)
Y0_axis = np.zeros(num_T)
Y1_axis = np.zeros(num_T)
Y2_axis = np.zeros(num_T) 
Y3_axis = np.zeros(num_T)
Y4_axis = np.zeros(num_T)
Y5_axis = np.zeros(num_T)

O1_axis = np.zeros(num_T)
O2_axis = np.zeros(num_T)

init_state = thermal_state/Z
noisy_init_state = np.load('new_init_state.npy')

Id= np.identity(dim)
Kick_Operator1 = (Id + 1j*A)/math.sqrt(2)
kicked_state1 = Kick_Operator1.dot((init_state).dot(np.conj(Kick_Operator1).T))
kicked_state2 = Kick_Operator1.dot((noisy_init_state).dot(np.conj(Kick_Operator1).T))

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

    w1 = mycode.trace(kicked_state1.dot(At_Ext)).real
    w2 = mycode.trace(kicked_state2.dot(At_Tro)).real


    T_axis[n] = t

    Y0_axis[n] = v0/Z
    Y1_axis[n] = v1/Z
    Y2_axis[n] = v2/Z
    Y5_axis[n] = mycode.trace(A.dot(At_Tro)).real/Z

    O1_axis[n] = w1
    O2_axis[n] = w2


plt.plot(T_axis, O1_axis,label = 'Im, Exact + Exact')
plt.plot(T_axis, O2_axis,label = 'Im, Noisy + Trotter')

plt.plot(T_axis, Y0_axis,label = 'Re, Exact + Exact')
plt.plot(T_axis, Y1_axis - Y2_axis + Y5_axis,label = 'Re, Noisy + Trotter')

np.save('T_axis.npy',T_axis)
np.save('signal_Im_ExtExt.npy',O1_axis)
np.save('signal_Im_NoiTro.npy',O2_axis)

np.save('signal_Re_ExtExt.npy',Y0_axis)
np.save('signal_Re_NoiTro.npy',Y1_axis - Y2_axis + Y5_axis)

plt.legend()
plt.show()



