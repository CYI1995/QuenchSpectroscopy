import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import source as mycode 

def Hankel(signal,M):

    H = np.zeros((M,M),dtype= complex)
    for i in range(M):
        for j in range(M):
            H[i][j] = signal[i+j] 

    return H 

def signal(frequencies, amplidtues, t):

    s = 0 + 1j*0
    L = len(frequencies)

    for l in range(L):
        s = s + amplidtues[l]*(math.cos(frequencies[l]*t) - 1j*math.sin(frequencies[l]*t))

    return s




def music(signal_list,w_min,w_max,dt,r,threshold):

    num_T = len(signal_list)

    T_list = np.zeros(num_T)
    for n in range(num_T):
        T_list[n] = dt*n 


    M = int(num_T/2)
    H = Hankel(signal_list,M)
    U,S,Vh = np.linalg.svd(H)
    for i in range(M):
        for j in range(r):
            U[i][j] = 0 


    M2 = int(10*M)
    w_list = np.zeros(M2)
    jw_list = np.zeros(M2)

    for j in range(M2):

        w = w_min + ((w_max - w_min)*dt)*j/M2

        phi_w = np.zeros(M,dtype=complex)
        for m in range(M):
            phi_w[m] = math.cos(w*m) - 1j*math.sin(w*m) 
        phi_w = phi_w/math.sqrt(M)
        Uhphiw = (np.conj(U).T).dot(phi_w)
        w_list[j] = w
        jw_list[j] = np.vdot(Uhphiw,Uhphiw).real  

    plt.plot(w_list/dt,jw_list)

    Output = []

    for i in range(1,M2-1):
        if(jw_list[i-1] > jw_list[i] and jw_list[i+1] > jw_list[i]):
            if(jw_list[i] < threshold):
                Output.append(w_list[i]/dt)

    return Output


