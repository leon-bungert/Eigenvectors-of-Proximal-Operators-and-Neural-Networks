'''
This script computes an eigenvector of the proximal operator of the total
variation using the proximal power method from: 

Leon Bungert, Ester Hait-Fraenkel, Nicolas Papadakis, and Guy Gilboa. 
"Nonlinear Power Method for Computing Eigenvectors of Proximal Operators 
and Neural Networks." arXiv preprint arXiv:2003.04595 (2020).

Please refer to this paper when you use the code.

authors: Leon Bungert <leon.bungert@fau.de>
date: 23.03.2021
'''

# standard
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
# ODL
import odl 
# custom
from algorithms import proximal_power_method, nossek_flow

# set up image domain
X = odl.uniform_discr([-1, -1], [1, 1], [100, 100], interp='linear')

# choose model and algorithm parameters
iso = False             # isotropic [True] or anisotropic [False] inner gradient norm
discr = 'forward'       # discretization of the gradient: 'forward', 'backward', 'central'
p_grad = 1              # outer gradient norm: total_variation: [p_grad=1] and H1 [p_grad=2]
sig = 0.05*1            # noise added to initial condition
save2disk = True        # save images to disk {True, False}
init = 'disk_ellipse'   # initial condition {'disk','square','noise','ellipse'}
i_max = 30              # maximum number of successful power iterations

#%% set up initial condition
if init == 'disk':
    data_func = lambda x: x[0]**2 + x[1]**2 < 0.7**2    
    data = X.element(data_func)
if init == 'disk_ellipse':
    data_func = lambda x: x[0]**2 + x[1]**2 < 0.7**2    
    data = X.element(data_func)
    data -= odl.phantom.geometric.defrise(X, nellipses=1)      
elif init == 'square': 
    data_func = odl.phantom.cuboid(X)
    data = X.element(data_func)
elif init == 'noise':
    data_func = odl.phantom.white_noise(X)
    data = X.element(data_func)
elif init == 'ellipse':
    data = odl.phantom.cuboid(X) - odl.phantom.geometric.defrise(X, nellipses=1)      
else:
    raise ValueError('Unknown initialization!')

# add noise
noise = odl.phantom.white_noise(X, mean=0, stddev = sig)
data += noise

# normalize data
H = odl.solvers.L2Norm(X)
proj = np.mean
data -= proj(data)
data /= H(data)
    

#%% set up functionals and operators
grad = odl.Gradient(X, method=discr, pad_mode='symmetric')
L = grad    
if discr == 'central':
    opNorm = np.sqrt(sum(1/X.cell_sides**2))
else:
    opNorm = 2*np.sqrt(sum(1/X.cell_sides**2))

L.norm = opNorm

if p_grad == 1:
    if iso is True:
        gradnorm = odl.solvers.GroupL1Norm(L.range, 2)
    else:
        gradnorm = odl.solvers.GroupL1Norm(L.range, 1)
elif p_grad == 2:    
    gradnorm = odl.solvers.L2Norm(L.range)

# define regularizer
g = gradnorm
J = g * L

#%% create output folder
if save2disk:
    folder = 'results/' + init
    if not os.path.exists(folder):
                os.makedirs(folder)
else:
    folder = None                    


affinities_cont = []
angles_cont = []

#%% run nossek flow
rules_nossek = ['Nossek']
if save2disk:
    folder_nossek = folder + '/nossek'
    if not os.path.exists(folder_nossek):
                os.makedirs(folder_nossek)
else:
    folder_nossek = None 
    
u, affs, angles = nossek_flow(data, J, H, folder=folder_nossek, proj=proj,\
                              save2disk=save2disk, i_max=i_max)
    
affinities_cont.append(affs)
angles_cont.append(angles)

#%% run power method
rules_power = ['Feld 0.01', 'Feld 0.1', 'Feld 1', 'Constant', 'Variable']

for rule in rules_power:
    if save2disk:
        folder_power = folder + '/power/' + rule
        if not os.path.exists(folder_power):
            os.makedirs(folder_power)
    else:
        folder_power = None 
        
    u, affs, angles = proximal_power_method(data, J, H, folder=folder_power,\
                                            proj=proj,save2disk=save2disk,\
                                            i_max=i_max, rule=rule)
    affinities_cont.append(affs)
    angles_cont.append(angles)


#%% Output settings
# use LaTeX fonts in the plot
plt.rc('text', usetex=True)

font = {'family' : 'serif',
        'size'   : 15}

matplotlib.rc('font', **font)


#%% Output only affinities
rules = rules_nossek + rules_power

markers = [None,None, None, None, None, None]
linestyles = ['-',\
              '-.',\
              '--',\
              '-',\
              '--',\
              '-']
    
colors = ['black', 'blue', 'blue', 'blue', 'red', 'red']
    
linewidth = 1.5

f, ax = plt.subplots()
ax.set_xlabel('Iterations')

ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0,decimals=0))
ax.set_xticks(range(0,i_max,5))

ax.set_ylabel('Affinity')  
for i in range(len(affinities_cont)):        
    ax.plot(affinities_cont[i], label=rules[i],color=colors[i],\
              marker=markers[i],\
              linestyle=linestyles[i],\
              linewidth=linewidth)
ax.tick_params(axis='y')
ax.legend(loc='lower right')
plt.tight_layout()      
     
if save2disk:   
    plt.savefig(folder + '/affinities.png',dpi=1000)

#%% Output only angles
rules = rules_nossek + rules_power

markers = [None,None, None, None, None, None]
linestyles = ['-',\
              '-.',\
              '--',\
              '-',\
              '--',\
              '-']
    
colors = ['black', 'blue', 'blue', 'blue', 'red', 'red']
    
linewidth = 1.5

f, ax = plt.subplots()
ax.set_xlabel('Iterations')

ax.set_xticks(range(0,i_max,5))

ax.set_ylabel('Angle')  
for i in range(len(angles_cont)):    
    angles_plot = [360/(2*np.pi) * np.arccos(1 - phi) for phi in angles_cont[i]]   
    ax.semilogy(angles_plot, label=rules[i],color=colors[i],\
              marker=markers[i],\
              linestyle=linestyles[i],\
              linewidth=linewidth)
ax.tick_params(axis='y')
ax.legend(loc='upper right')
plt.tight_layout()      
     
if save2disk:   
    plt.savefig(folder + '/angles.png',dpi=1000)    

        
        
