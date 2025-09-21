import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random

import scipy.linalg
import source as mycode
import statistics

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

def JordanWinger_neighbor_hopping(i,num):


    XX = mycode.XX_pair(num,i,i+2)
    YY = mycode.YY_pair(num,i,i+2)
    Z = mycode.SingleZ(i+1,num)

    return 0.5*(XX + YY).dot(Z)

def JordanWigner_ni(i,num):

    dim = 2**num 

    return 0.5*(np.identity(dim) - mycode.SingleZ(i,num))

import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random

import scipy.linalg
import source as mycode
import statistics

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

def JordanWigner_ni(i,num):

    dim = 2**num 

    return 0.5*(np.identity(dim) - mycode.SingleZ(i,num))

print('Running')

num_sites = 4
dim = 2**num_sites

t = 1
U = 2

ham0 = np.zeros((dim,dim),dtype = complex)

c0 = JordanWigner_ci(0,num_sites)
c1 = JordanWigner_ci(1,num_sites)
c2 = JordanWigner_ci(2,num_sites)
c3 = JordanWigner_ci(3,num_sites)


ham0 = ham0 - t*((np.conj(c0).T).dot(c2))

ham0 = ham0 - t*((np.conj(c1).T).dot(c3))


ham0 = ham0 + np.conj(ham0).T

for j in range(int(num_sites/2)):
    n2j = JordanWigner_ni(2*j,num_sites) 
    n2jp1 = JordanWigner_ni(2*j+1,num_sites) 
    ham0 = ham0 - U*n2j.dot(n2jp1)

A0 = (c0 + np.conj(c0).T)/2


Sum_Z = np.zeros((dim,dim),dtype = complex)
for j in range(num_sites):
    Sum_Z = Sum_Z + mycode.SingleZ(j,num_sites)
Parity = scipy.linalg.expm(1j*math.pi*Sum_Z/2)

norm = mycode.matrix_norm(ham0)
ham = math.pi*ham0/norm



ham_sym = 0.5 * (ham + ham.dot(Parity))
ham_asym = 0.5 * (ham - ham.dot(Parity))


beta = 1
thermal_state = scipy.linalg.expm(-beta*ham)
thermal_state_sym = scipy.linalg.expm(-beta*ham_sym)
thermal_state_asym = scipy.linalg.expm(-beta*ham_asym)

np.save('ham.npy',ham)
np.save('ham_sym.npy',ham_sym)
np.save('ham_asym.npy',ham_asym)
np.save('A0.npy',A0)
np.save('Parity.npy',Parity)
np.save('thermal_state.npy',thermal_state)
np.save('thermal_state_sym.npy',thermal_state_sym)
np.save('thermal_state_asym.npy',thermal_state_asym)

# plt.plot(np.zeros(dim),parity_list - parity_list[0],marker = 'x')
# plt.scatter(amplidtues1,parity_list - parity_list[0],marker = '.')
# plt.scatter(energy_gap,amplidtues1,marker = 'x')
# plt.scatter(energy_gap,amplidtues2,marker = 'x')
# plt.show()