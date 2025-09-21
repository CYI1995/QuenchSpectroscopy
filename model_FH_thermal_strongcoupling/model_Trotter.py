import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random

import scipy.linalg
import source as mycode
import statistics
import cmath

def h(eps,k_temp,t):

    return eps*math.cos(k_temp*t)

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

def JordanWigner_neighbor_hopping(i,num):


    XX = mycode.XX_pair(num,i,i+2)
    YY = mycode.YY_pair(num,i,i+2)
    Z = mycode.SingleZ(i+1,num)

    return 0.5*(XX + YY).dot(Z)

def JordanWigner_ni(i,num):

    dim = 2**num 

    return 0.5*(np.identity(dim) - mycode.SingleZ(i,num))


num_sites = 12
dim = 2**num_sites

t = 1
U = 6

ham0 = np.zeros((dim,dim),dtype = complex)

c0 = JordanWigner_ci(0,num_sites)
c1 = JordanWigner_ci(1,num_sites)
c2 = JordanWigner_ci(2,num_sites)
c3 = JordanWigner_ci(3,num_sites)
c4 = JordanWigner_ci(4,num_sites)
c5 = JordanWigner_ci(5,num_sites)
c6 = JordanWigner_ci(6,num_sites)
c7 = JordanWigner_ci(7,num_sites)
c8 = JordanWigner_ci(8,num_sites)
c9 = JordanWigner_ci(9,num_sites)
c10 = JordanWigner_ci(10,num_sites)
c11 = JordanWigner_ci(11,num_sites)


T0 = np.zeros((dim,dim),dtype = complex)
T1 = np.zeros((dim,dim),dtype = complex)
T2 = np.zeros((dim,dim),dtype = complex)
T3 = np.zeros((dim,dim),dtype = complex)

T0 = T0 - t*((np.conj(c0).T).dot(c2))
T0 = T0 - t*((np.conj(c1).T).dot(c3))
T0 = T0 - t*((np.conj(c6).T).dot(c8))
T0 = T0 - t*((np.conj(c7).T).dot(c9))
T0 = T0 + np.conj(T0).T 


T1 = T1 - t*((np.conj(c2).T).dot(c4))
T1 = T1 - t*((np.conj(c3).T).dot(c5))
T1 = T1 - t*((np.conj(c8).T).dot(c10))
T1 = T1 - t*((np.conj(c9).T).dot(c11))
T1 = T1 + np.conj(T1).T

T2 = T2 -  t*((np.conj(c0).T).dot(c6))
T2 = T2 -  t*((np.conj(c1).T).dot(c7))
T2 = T2 -  t*((np.conj(c2).T).dot(c8))
T2 = T2 -  t*((np.conj(c3).T).dot(c9))
T2 = T2 -  t*((np.conj(c4).T).dot(c10))
T2 = T2 -  t*((np.conj(c5).T).dot(c11))
T2 = T2 + np.conj(T2).T

for j in range(int(num_sites/2)):
    n2j = JordanWigner_ni(2*j,num_sites) 
    n2jp1 = JordanWigner_ni(2*j+1,num_sites) 
    T3 = T3 - U*n2j.dot(n2jp1)

ham0 = T0 + T1 + T2 + T3
norm = mycode.matrix_norm(ham0)

T0 = math.pi*T0/norm
T1 = math.pi*T1/norm 
T2 = math.pi*T2/norm 
T3 = math.pi*T3/norm
ham = math.pi*ham0/norm
A0 = (c0 + np.conj(c0).T)/2

Sum_Z = np.zeros((dim,dim),dtype = complex)
for j in range(num_sites):
    Sum_Z = Sum_Z + mycode.SingleZ(j,num_sites)
Parity = scipy.linalg.expm(1j*math.pi*Sum_Z/2)

ham_sym = 0.5 * (ham + ham.dot(Parity))
ham_asym = 0.5 * (ham - ham.dot(Parity))

beta = 1
print('thermal state')
thermal_state = scipy.linalg.expm(-beta * ham)
print('thermal state sym')
thermal_state_sym = scipy.linalg.expm(-beta * ham_sym)
print('thermal state asym')
thermal_state_asym = scipy.linalg.expm(-beta * ham_asym)

np.save('T0.npy',T0)
np.save('T1.npy',T1)
np.save('T2.npy',T2)
np.save('T3.npy',T3)
np.save('ham.npy',ham)
np.save('ham_sym.npy',ham_sym)
np.save('ham_asym.npy',ham_asym)
np.save('A0.npy',A0)
np.save('Parity.npy',Parity)
np.save('thermal_state.npy',thermal_state)
np.save('thermal_state_sym.npy',thermal_state_sym)
np.save('thermal_state_asym.npy',thermal_state_asym)






