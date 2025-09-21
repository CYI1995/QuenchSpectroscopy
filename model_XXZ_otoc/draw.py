import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import string
import matplotlib as mpl
import source as mycode

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

mpl.rcParams.update(mpl.rcParamsDefault)

def sparse(array,L,k):

    M = int(L/k)
    new_array = np.zeros(M)
    for j in range(M):
        idx = (j+1)*k
        new_array[j] = array[idx-1]
    return new_array

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

font = {'family' : 'normal',
         'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 




T_list = np.load('T_axis.npy')
T_list2 = np.load('T_axis2.npy')
signalA= np.load('signal_Im_ExtExt.npy')
signalB= np.load('signal_Im_NoiTro2.npy')
signalC= np.load('signal_Im_NoiTro.npy')
signalD= np.load('signal_Re_ExtExt.npy')
signalE= np.load('signal_Re_NoiTro2.npy')
signalF= np.load('signal_Re_NoiTro.npy')


print(signalB)

fig, axs = plt.subplot_mosaic([['(a)'], ['(b)']],
                              constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')
    if(label == '(a)'):
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'Im [OTOC$(A,B,t)]$')
        ax.plot(T_list,signalA,label = 'Exact + Exact')
        # ax.plot(T_list,signalC,linestyle = 'dashed',label = r'Noisy + Trotter, $dt = \pi/100$')
        ax.plot(T_list2,signalB,linestyle = 'dashed',label = r'Noisy + Trotter')
        ax.set_xlim(0,50*math.pi)
        ax.legend(loc = 1)
    if(label == '(b)'):
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'Re [OTOC$(A,B,t)$]')
        ax.plot(T_list,signalD,label = 'Exact + Exact')
        # ax.plot(T_list,signalF,linestyle = 'dashed',label = r'Noisy + Trotter, $dt = \pi/100$')
        ax.plot(T_list2,signalE,linestyle = 'dashed',label = r'Noisy + Trotter')
        ax.set_xlim(0,50*math.pi)
        ax.legend(loc = 1)
plt.show()