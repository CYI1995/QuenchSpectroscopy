import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random

import scipy.linalg
import source as mycode
import statistics
import cmath

def Hankel(signal,M):

    H = np.zeros((M,M),dtype= complex)
    for i in range(M):
        for j in range(M):
            H[i][j] = signal[i+j] 

    return H 


def music(signal_list,w_min,w_max,dt):

    num_T = len(signal_list)

    T_list = np.zeros(num_T)
    for n in range(num_T):
        T_list[n] = dt*n 


    M = int(num_T/2)
    H = Hankel(signal_list,M)
    U,S,Vh = np.linalg.svd(H)

    for j in range(M):
        if(S[j] > 1):
            print(j)
            for i in range(M):
                U[i][j] = 0 


    M2 = int(10*M)
    w_list = np.zeros(M2)
    jw_list = np.zeros(M2)

    for j in range(M2):

        w = w_min*dt + ((w_max - w_min)*dt)*j/M2

        phi_w = np.zeros(M,dtype=complex)
        for m in range(M):
            phi_w[m] = cmath.exp(1j*w*m) 

        phi_w = phi_w/math.sqrt(M)
        Uhphiw = (np.conj(U).T).dot(phi_w)
        w_list[j] = w
        jw_list[j] = np.vdot(Uhphiw,Uhphiw).real  

    return w_list/dt,jw_list


def extract_amplitudes(signal,frequencies,dt):

    L1 = len(frequencies)
    L2 = len(signal) 

    Jump = int(L2/L1)

    Kernel = np.zeros((L1,L1))
    SignalOnKernel = np.zeros(L1)
    for i in range(L1):
        t_temp = (Jump*i+1)*dt
        SignalOnKernel[i] = signal[Jump*i+1]
        for j in range(L1):
            Kernel[i][j] = math.sin(frequencies[j]*t_temp)

    Amplitues = np.linalg.inv(Kernel).dot(SignalOnKernel)

    return Amplitues

def signal(frequencies, amplitudes,t):

    s = 0 + 1j*0 
    L = len(frequencies)

    for l in range(L):
        s = s + amplitudes[l] * cmath.exp(1j*frequencies[l] * t)

    return s

        

num_T = 4000
nT = 200
T= nT*math.pi
dt = T/num_T

signalA= np.load('signal_Im_ExtExt.npy')
signalC= np.load('signal_Im_NoiTro.npy')
signalD= np.load('signal_Re_ExtExt.npy')
signalF= np.load('signal_Re_NoiTro.npy')

w_list,jw_list = music(1j * signalA + signalD,-math.pi/2,math.pi/2,dt)
np.save('w_list_extext.npy',w_list)
np.save('jw_list_extext.npy',jw_list)
# w_list = np.load('w_list.npy')
# jw_list = np.load('jw_list.npy')
plt.plot(w_list,jw_list,c = 'k',label = 'ExtExt')


w_list,jw_list = music(1j * signalC + signalF,-math.pi/2,math.pi/2,dt)
np.save('w_list_noitro.npy',w_list)
np.save('jw_list_noitro.npy',jw_list)
# w_list = np.load('w_list.npy')
# jw_list = np.load('jw_list.npy')
plt.plot(w_list,jw_list,c = 'b',linestyle = '--',label = 'NoiTro')

plt.legend()
plt.show()


# Output = []

# for i in range(1,len(jw_list)-1):
#     if(jw_list[i-1] > jw_list[i] and jw_list[i+1] > jw_list[i]):
#         if(jw_list[i] < 0.1):
#             Output.append(w_list[i])

# np.save('frequencies.npy',Output)
# # Output = np.load('frequencies.npy')

# L = len(Output)

# SinMatrix = np.zeros((L,num_T))
# for nt in range(num_T):
#     t = (nt + 1)*dt 
#     for k in range(L):
#         Ek = Output[k]
#         SinMatrix[k][nt] = math.sin(Ek*t)

# SinSin = SinMatrix.dot(SinMatrix.T)
# signalSin = signal_A.dot(SinMatrix.T)
# Amp_list = signalSin.dot(np.linalg.inv(SinSin))

# print(Amp_list)
# np.save('alphan.npy',Amp_list)


# w_list,jw_list = music(signal_B,0,2*math.pi,dt,10)
# Output = []

# for i in range(1,len(jw_list)-1):
#     if(jw_list[i-1] > jw_list[i] and jw_list[i+1] > jw_list[i]):
#         if(jw_list[i] < 0.1):
#             Output.append(w_list[i])

# L = len(Output)
# SinMatrix = np.zeros((L,num_T))
# for nt in range(num_T):
#     t = (nt + 1)*dt 
#     for k in range(L):
#         Ek = Output[k]
#         SinMatrix[k][nt] = math.sin(Ek*t)

# SinSin = SinMatrix.dot(SinMatrix.T)
# signalSin = signal_A.dot(SinMatrix.T)
# Amp_list = signalSin.dot(np.linalg.inv(SinSin))

# print(Amp_list)
# np.save('betan.npy',Amp_list)



