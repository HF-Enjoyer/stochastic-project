import numpy as np
from matplotlib import pyplot as plt
#from autograd import grad
#from autograd import hessian
#import autograd.numpy as anp
from wavefunction import *

def random_walker(a1, step_size, N_steps):
    # Initialization of starting positions of electrons
    r1 = np.random.normal(0.53/2, 1, size=3)
    r2 = (-1)*r1

    trajectory1 = [r1]
    trajectory2 = [r2]

    for _ in range(N_steps):
        i = np.random.choice([1, 2]) # Choose r1 or r2 electron
        if i == 1:
            new_r1 = r1 + np.random.uniform(-step_size, step_size, size=3)
            p = min(1, np.abs(vmc_WF(a1, new_r1, r2)/vmc_WF(a1, r1, r2))**2)
            if p >= np.random.uniform(0, 1):
                r1 = new_r1
        elif i == 2:
            new_r2 = r2 + np.random.uniform(-step_size, step_size, size=3)
            p = min(1, np.abs(vmc_WF(a1, r1, new_r2)/vmc_WF(a1, r1, r2))**2)
            if p >= np.random.uniform(0, 1):
                r2 = new_r2
        trajectory1.append(r1.copy())
        trajectory2.append(r2.copy())
    
    return np.array(trajectory1), np.array(trajectory2)

def analytical_en_loc(beta, r1, r2):

    r12 = np.linalg.norm(r1-r2, ord=2)
    r1_norm = np.linalg.norm(r1, ord=2)
    r2_norm = np.linalg.norm(r2, ord=2)
    Z = 2
    E_L1 = 1/r12 - Z**2
    factor1 = 1/(2*(1+beta*r12)**2)
    factor_with_dot = 1-np.dot(r1, r2)/(r1_norm*r2_norm)
    term2 = factor1*((Z*(r1_norm+r2_norm)/(r12))*factor_with_dot - factor1 -2/r12 + 2*beta/(1+beta*r12))
    return E_L1 + term2

# DO NOT USE
def local_energy(beta, r1, r2): 
    # initialize values
    r12 = anp.linalg.norm(r1-r2, ord=2)

    hessian_WF_r1 = hessian(vmc_WF, argnum=1)
    hessian_WF_r2 = hessian(vmc_WF, argnum=2)
    kinetic_energy = (1/vmc_WF(beta, r1, r2))*(-0.5*anp.trace(hessian_WF_r1(beta, r1, r2))-0.5*anp.trace(hessian_WF_r2(beta, r1, r2)))

    r1 = anp.linalg.norm(r1, ord=2)
    r2 = anp.linalg.norm(r2, ord=2)
    # V(R) part - direct...
    E_pot_part = -2/r1 -2/r2 + 1/r12

    return kinetic_energy + E_pot_part

def variational_energy(beta, tr1, tr2, N_eq, analytical=True):
    tr1_eq = tr1[N_eq:]
    tr2_eq = tr2[N_eq:]
    E_loc_all = 0
    if analytical == True:
        en_loc = analytical_en_loc
    else:
        en_loc = local_energy
    for i in range(len(tr1_eq)):
        r1 = tr1_eq[i]
        r2 = tr2_eq[i]
        E_loc_all += en_loc(beta, r1, r2)
    return E_loc_all/len(tr1_eq)
