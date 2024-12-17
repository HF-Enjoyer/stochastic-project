import numpy as np
from matplotlib import pyplot as plt
#from autograd import grad
#from autograd import hessian
#import autograd.numpy as anp
import tqdm
from wavefunction import *
from test import *
from tqdm import tqdm
from scipy.optimize import minimize

def average_over_trajectories(beta, N_trajectories, step_size, N_steps, N_eq):
    energies = []
    for _ in range(N_trajectories):
        tr1, tr2 = random_walker(beta, step_size, N_steps)
        energy = variational_energy(beta, tr1, tr2, N_eq)
        energies.append(energy)
    return np.mean(energies) 

def main():
    N_trjs = 20
    dx = 0.1
    N_steps = 15000
    N_eq = 5000

    betas = np.linspace(0.01, 3.01, num=25)
    var_ens = []

    for beta in tqdm(betas):
        energy = average_over_trajectories(beta, N_trjs, dx, N_steps, N_eq)
        var_ens.append(energy)
        with open('result.txt', 'a') as f1:
            f1.write(f'beta = {beta} \t Energy = {energy}\n')
    print("Energy minima = ", min(var_ens), "Hartree")

    plt.plot(betas, var_ens, '.-', color='orange')
    plt.title(r'$\text{Energy}(\beta)$')
    plt.grid()
    plt.savefig('result.png', dpi=300, bbox_inches='tight')

if __name__=="__main__":
    main()
