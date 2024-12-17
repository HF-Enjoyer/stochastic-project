import numpy as np
from matplotlib import pyplot as plt
#import torch
#from autograd import grad
#from autograd import hessian
#import autograd.numpy as anp

# Every calculation is in BOHR units (a. u.) e.g. for length 1 a.u. = 0.53 Angstrom

def HF_wavefunction(r1, r2):
    '''This is the expression for He atom WF with Z = 2
    All computations are in bohr units, so a_0 = 1 bohr
    r1, r2 are 3D arrays of electron coordinates'''
    r1 = np.linalg.norm(r1, 2)
    r2 = np.linalg.norm(r2, 2)

    return (2**3)/(np.pi)*np.exp(-2*(r1+r2)) # Z = 2 is nucleus charge

def Jastrow(beta, r1, r2):
    '''This is the Jastrow factor J, the exact WF Psi is HF*J.
    This factor accounts correlation between two electrons with coordinates r1 and r2.
    '''
    r_12 = np.linalg.norm(r1-r2, ord=2)
    u_12 = r_12/(2*(1 + beta*r_12))
    return np.exp(u_12) # Check carefully if it is really needed to multiply by -1

# Returns the product of factor J and wavefunction HF
def vmc_WF(beta, r1, r2):
    return Jastrow(beta, r1, r2)*HF_wavefunction(r1, r2)
