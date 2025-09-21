import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
import random

import scipy.linalg
import source as mycode
import statistics
import cmath

ham = np.load('ham.npy')
ham_sym = np.load('ham_sym.npy')
ham_asym = np.load('ham_asym.npy')

P = np.load('Parity.npy')
thermal_state = np.load('thermal_state.npy')
thermal_state_sym = np.load('thermal_state_sym.npy')
thermal_state_asym = np.load('thermal_state_asym.npy')

d = len(ham[0])
Z = mycode.trace(thermal_state)
ZS = mycode.trace(thermal_state_sym) - 0.5*d
ZA = mycode.trace(thermal_state_asym) - 0.5*d

print(ZS + ZA - Z)


parity_sym_sum = mycode.trace(thermal_state_sym.dot(P))
parity_asym_sum = mycode.trace(thermal_state_asym.dot(P))

NA = ZS - parity_sym_sum 
NS = ZA + parity_asym_sum 

Parameters = np.array([ZS,ZA,NS,NA])
print(Parameters)
np.save('Parameters.npy',Parameters)