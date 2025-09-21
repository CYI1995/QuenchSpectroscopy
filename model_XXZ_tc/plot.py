import numpy  as np
import scipy
import math
from scipy import linalg
import matplotlib
from matplotlib import pyplot as plt
import string
import matplotlib as mpl

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
        # 'weight' : 'bold',
        'size'   : 18}

matplotlib.rc('font', **font)
matplotlib.rc('xtick', labelsize=18) 
matplotlib.rc('ytick', labelsize=18) 

SAP = np.load('signal_analysis_parameters.npy')
num_t = SAP[0]
nT = SAP[1]
T = nT*math.pi 
T_list = np.linspace(0,T,num_t)
T_list2 = np.linspace(0,10*T,10*num_t)

Signal1 = np.load('signal_A_Tro.npy')
Signal2 = np.load('signal_A.npy')

plt.plot(T_list,Signal1)
plt.plot(T_list,Signal2)
plt.show()

# energy_data = np.load('energy_gap.npy')
# amplitude_data = np.load('amplitude.npy')  
# amplitude_data = amplitude_data/np.max(amplitude_data)
# w_list = np.load('w_list.npy')
# jw_list = np.load('jw_list.npy')
# M_data = np.load('M_data.npy')
# frequencies = np.load('frequencies.npy')
# L = len(frequencies)
# y = np.linspace(-0.1,1.1,100)



# # fig, axs = plt.subplot_mosaic([['(a)','(b)','(c)']],
# #                               constrained_layout=True)

# fig, axs = plt.subplot_mosaic([['(a)'], ['(b)'],['(c)']],
#                               constrained_layout=True)

# for label, ax in axs.items():
#     # label physical distance to the left and up:
#     trans = matplotlib.transforms.ScaledTranslation(-20/72, 7/72, fig.dpi_scale_trans)
#     ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
#             fontsize='medium', va='bottom', fontfamily='serif')
    
#     if(label == '(a)'):
#         ax.set_xlabel(r'$t$')
#         ax.set_ylabel(r'$\Delta Q(t)$')

#         ax.plot(T_list,Signal1,c="k",label = r'$(O_p + O_p^\dagger)/2$')
#         ax.plot(T_list,Signal2,c='b',linestyle = 'dotted',label = r'$(O_p - O_p^\dagger)/(2i)$')
#         ax.set_ylim(-3,5)
#         ax.legend(loc = 1)

#     if(label == '(b)'):
#         ax.set_xlabel(r'$k$')

#         ax.plot(w_list,jw_list,c="k",label = r'$J(\omega)$')
#         ax.scatter(energy_data,amplitude_data,c="r",marker ='x')

#         for i in range(L):
#             k = frequencies[i]
#             ax.plot(k*np.ones(100),y,linestyle = '--',label = r'$k$ = %.4f' %k)

#         ax.set_xlim(0,math.pi/4)
#         ax.set_ylim(-0.1,1.1)
#         ax.set_xticks(np.arange(0, 0.25*np.pi+0.01, np.pi/16))
#         labels = ['$0$', r'$\pi/16$', r'$\pi/8$', r'$3\pi/16$', r'$\pi/4$']
#         ax.set_xticklabels(labels)
#         ax.legend(loc = 1) 
#     if(label == '(c)'):
#         ax.set_xlabel(r'$t$')
#         ax.set_ylabel(r'$\Delta M(k,t)$')
#         for i in range(L-1):
#             k = frequencies[i]
#             Delta_M1 = M_data[i]
#             Delta_M2 = M_data[i+L]
#             ax.plot(T_list2,-Delta_M1)
#             ax.plot(T_list2,-Delta_M2,linestyle = 'dotted')

#         k = frequencies[L-1]
#         Delta_M1 = M_data[L-1]
#         Delta_M2 = M_data[2*L-1]
#         ax.plot(T_list2,-Delta_M1,label = r'$(O_p + O_p^\dagger)/2$')
#         ax.plot(T_list2,-Delta_M2,linestyle = 'dotted',label = r'$(O_p - O_p^\dagger)/(2i)$')

#         ax.set_ylim(-0.1,3)
#         ax.legend(loc = 2)


# plt.show()