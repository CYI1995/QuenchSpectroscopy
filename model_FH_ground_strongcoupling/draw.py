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
signalA= np.load('signal_Im_ExtExt.npy')
signalC= np.load('signal_Im_NoiTro.npy')
signalD= np.load('signal_Re_ExtExt.npy')
signalF= np.load('signal_Re_NoiTro.npy')


Amplitudes = np.load('Amplitudes_GS.npy')
Energy_gaps = np.load('Energy_gap_GS.npy')
Amplitudes = Amplitudes/np.max(Amplitudes)

w0_list = np.load('w_list_extext.npy')
jw0_list = np.load('jw_list_extext.npy')
w2_list = np.load('w_list_noitro.npy')
jw2_list = np.load('jw_list_noitro.npy')

fig, axs = plt.subplot_mosaic([['(a)'], ['(b)'],['(c)']],
                              constrained_layout=True)

for label, ax in axs.items():
    # label physical distance to the left and up:
    trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
    ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='medium', va='bottom', fontfamily='serif')
    if(label == '(a)'):
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'Im $[C(A,A,t)]$')
        ax.plot(T_list,signalA,label = 'Exact + Exact')
        ax.plot(T_list,signalC,linestyle = 'dashed',label = 'Noisy + Trotter')
        ax.set_xlim(0,50*math.pi)
        ax.legend(loc = 1)
    if(label == '(b)'):
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'Re $[C(A,A,t)]$')
        ax.plot(T_list,signalD,label = 'Exact + Exact')
        ax.plot(T_list,signalF,linestyle = 'dashed',label = 'Noisy + Trotter')
        ax.set_xlim(0,50*math.pi)
        ax.legend(loc = 1)
    if(label == '(c)'):
        ax.set_xlabel(r'$\omega$')
        ax.set_ylabel(r'$R(\omega)$')
        ax.plot(w0_list,jw0_list,c = 'k',label = 'Exact + Exact')
        ax.plot(w2_list,jw2_list,c = 'b',linestyle = 'dashed',label = 'Noisy + Trotter')
        ax.scatter(-Energy_gaps,Amplitudes,marker = 'x',label = 'Excitation spectrum')
        ax.set_xticks(np.arange(0, np.pi/2+0.01, np.pi/8))
        labels = [r'$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$']
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.01, math.pi/2 + 0.01)
        ax.set_ylim(-0.1,2.1)
        ax.legend(loc = 1)
plt.show()