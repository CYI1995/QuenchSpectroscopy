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
B = np.load('B.npy')
P = np.load('Parity.npy')
thermal_state = np.load('thermal_state.npy')
thermal_state_sym = np.load('thermal_state_sym.npy')
thermal_state_asym = np.load('thermal_state_asym.npy')

TX = np.load('TX.npy')
TY = np.load('TY.npy')
TZ = np.load('TZ.npy')

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
I0_axis = np.zeros(num_T)
I1_axis = np.zeros(num_T)
R0_axis = np.zeros(num_T)
R1_axis = np.zeros(num_T)
R2_axis = np.zeros(num_T) 
R3_axis = np.zeros(num_T)
init_state = thermal_state/Z
init_state_sym = thermal_state_sym/(ZS + NA)
init_state_asym = thermal_state_asym/(ZA + NS)

noisy_init_state = np.load('new_init_state.npy')



Id= np.identity(dim)
Kick_Operator1 = (Id + 1j*A)/math.sqrt(2)
kicked_state1 = Kick_Operator1.dot((init_state).dot(np.conj(Kick_Operator1).T))
kicked_state2 = Kick_Operator1.dot((noisy_init_state).dot(np.conj(Kick_Operator1).T))


noisy_init_state_sym = np.load('new_init_state_sym.npy')
noisy_init_state_asym = np.load('new_init_state_asym.npy')

Kick_Operator2 = (P + A)/math.sqrt(2)
kicked_state_sym = Kick_Operator2.dot((init_state_sym).dot(np.conj(Kick_Operator2).T))
kicked_state_asym = Kick_Operator2.dot((init_state_asym).dot(np.conj(Kick_Operator2).T))

U_Tro_dt = scipy.linalg.expm(-1j*TZ*dt)
U_Tro_dt = scipy.linalg.expm(-1j*TY*dt).dot(U_Tro_dt)
U_Tro_dt = scipy.linalg.expm(-1j*TX*dt).dot(U_Tro_dt)

U_Ext_dt = scipy.linalg.expm(-1j*ham*dt)

Bt_temp_Ext = B.copy()
Bt_temp_Tro = B.copy()

for n in range(num_T):

    print(n)

    t = (n+1)*dt
    Bt_temp_Ext = (np.conj(U_Ext_dt).T).dot(Bt_temp_Ext.dot(U_Ext_dt))
    AtA_Ext = Bt_temp_Ext.dot(A.dot(Bt_temp_Ext))

    Bt_temp_Tro = (np.conj(U_Tro_dt).T).dot(Bt_temp_Tro.dot(U_Tro_dt))
    AtA_Tro = Bt_temp_Tro.dot(A.dot(Bt_temp_Tro))

    w1 = mycode.trace(kicked_state1.dot(AtA_Ext)).real
    w2 = mycode.trace(kicked_state2.dot(AtA_Tro)).real

    v1 = Z * mycode.trace(init_state.dot(A.dot(AtA_Ext))).real
    v2 = (NA + ZS)* mycode.trace(kicked_state_sym.dot(AtA_Tro)).real
    v3 = (NS + ZA)* mycode.trace(kicked_state_asym.dot(AtA_Tro)).real

    T_axis[n] = t

    I0_axis[n] = w1
    I1_axis[n] = w2

    T_axis[n] = t
    R0_axis[n] = v1/Z
    R1_axis[n] = v2/Z
    R2_axis[n] = v3/Z
    R3_axis[n] = mycode.trace(A.dot(AtA_Tro)).real/Z


plt.plot(T_axis, I0_axis,label = 'Im, Exact + Exact')
plt.plot(T_axis, I1_axis,label = 'Im, Noisy + Trotter')

plt.plot(T_axis, R0_axis,label = 'Re, Exact + Exact')
plt.plot(T_axis, R1_axis - R2_axis + R3_axis, label = 'Re, Noisy + Trotter')

np.save('T_axis2.npy',T_axis)
# np.save('signal_Im_ExtExt.npy',I0_axis)
np.save('signal_Im_NoiTro2.npy',I1_axis)

# np.save('signal_Re_ExtExt.npy',R0_axis)
np.save('signal_Re_NoiTro2.npy',R1_axis - R2_axis + R3_axis)

plt.legend()
plt.show()



