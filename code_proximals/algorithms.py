'''
This module implements the proximal power method from: 

Leon Bungert, Ester Hait-Fraenkel, Nicolas Papadakis, and Guy Gilboa. 
"Nonlinear Power Method for Computing Eigenvectors of Proximal Operators 
and Neural Networks." arXiv preprint arXiv:2003.04595 (2020).

Please refer to this paper when you use the code.

For comparison this module also implements the flow from:
Raz Z. Nossek, and Guy Gilboa. "Flows generating nonlinear eigenfunctions." 
Journal of Scientific Computing 75, no. 2 (2018): 859-888.

authors: Leon Bungert <leon.bungert@fau.de>
date: 23.03.2021
'''

import numpy as np
import odl
import matplotlib.pyplot as plt

##############################################################################
def proximal_power_method(data, J, H, folder=None, proj=None, save2disk=False,\
                          i_max=10, rule='Constant'):
    
    X = data.space
    g = J.left
    L = J.right
    
    if proj is None:
        proj = odl.ZeroOperator(X)
    
    # get primal dual step sizes
    (tau, sigma) = odl.solvers.pdhg_stepsize(L.norm)
    
    #%% intialize
    
    # initialize iterates and eigenvalues
    ev_m = np.inf
    data -= proj(data)
    data /= H(data)
    u = data.copy()
    um = u.copy()
    
    # choose step size constant 0 < c < 1
    c = 0.9
    
    # plot initialization
    u.show('Iteration ' + str(0))
    
    
    #
    rules = rule.split(' ')
    
    # define step size
    if rules[0] == 'Constant':
        alpha = c * H(um-proj(um))/J(um)
    
    # initialize iteration counter and output vectors
    i = 0
    affs = []
    angles = []
    
    # output 
    if save2disk and folder is not None:        
        filename = '_it_' + str(i)
        plt.imsave(folder + '/' + filename + '.png', u, cmap='gray')
    
    #%% run power method
    it = 0  # couter for proximal backtracking
    it_max = 10 # maximum number of prox evaluations
    while i < i_max:
    
        # define variable step size
        if rules[0] == 'Variable':
            alpha = c /J(um)
        elif rules[0] == 'Feld':
            if len(rules) > 1:
                dt = float(rules[1])
            else:
                dt = 1.
            alpha = c * dt / (1 + dt * J(um)) 
            
        # define data fidelity    
        f = (0.5/alpha) * odl.solvers.L2NormSquared(X).translated(um)
        
        # evaluate the proximal operator    
        odl.solvers.pdhg(u,f,g,L,1000,tau,sigma) 
        
        # compute subgradient
        sg = (um - u)/alpha    
       
        # subtract mean to be on the safe side
        u -= proj(u)
        
        # evaluate proximal eigenvalue
        ev_prox = u.inner(um)/(H(um)**2)
          
        # normalize to create the iterate
        u /= H(u)
        
        # evaluate Rayleigh quotient
        ev = J(u)/H(u-proj(u))
    
        # check whether Rayleigh quotient decreased
        if ev < ev_m:
            it = 0
                        
            # compute eigenvector affinity in <= 1
            aff = odl.solvers.L2NormSquared(X)(sg)/J(sg)
            affs.append(aff)
            
            # evaluate angle            
            angle = 1 - um.inner(u)/(H(u)*H(um))
            angles.append(angle)
                
            # output
            print(20*'<>')                      
            print('Affinity: {:.2f}'.format(aff))
            print('Angle: {:.2E} '.format(angle))
            print('Subdiff Eigenvalue: {:.2f} '.format(ev))
            print('Prox Eigenvalue: {:.2f} '.format(ev_prox))
            print('Theoretical Prox Eigenvalue: {:.2f} '.format(1-alpha*ev))
            print(20*'<>')                      
        
            # update iteration counter and plot image
            i += 1
            u.show('Iteration ' + str(i))
            
            # save result to disk
            if save2disk and folder is not None:        
                filename = 'it_' + str(i)
                plt.imsave(folder + '/' + filename + '.png', u, cmap='gray')
                
            # update old variables    
            um = u.copy()
            ev_m = ev
        else:
            it += 1
            c *= 0.9
            
            if it >= it_max: 
                
                #evaluate angle                
                tmp = um.copy()
                f = (0.5/alpha) * odl.solvers.L2NormSquared(X).translated(tmp)        
                odl.solvers.pdhg(um,f,g,L,1000,tau,sigma) 
                angle = 1-um.inner(tmp)/(H(um)*H(tmp))
                angles.append(angle)
                
                #evaluate affinity
                sg = (um-tmp)/alpha
                aff = odl.solvers.L2NormSquared(X)(sg)/J(sg)
                affs.append(aff)
                                                                           
                print('Stopped iterations because Rayleigh quotient cannot be improved')
                break
            else:
                continue
            
    return um, affs, angles     
##############################################################################        
def nossek_flow(data, J, H, folder=None, proj=None, \
                save2disk=False, i_max=10):
    
    X = data.space
    g = J.left
    L = J.right
    
    if proj is None:
        proj = odl.ZeroOperator(X)
    
    # get primal dual step sizes
    (tau, sigma) = odl.solvers.pdhg_stepsize(L.norm)
    
    #%% intialize
    
    # initialize iterates and eigenvalues
    ev_m = np.inf
    data -= proj(data)
    data /= H(data)
    u = data.copy()
    um = u.copy()
    
    # choose step size constant 0 < c < 1
    c = 0.9
    
    # plot initialization
    u.show('Iteration ' + str(0))
 
    dt = c * H(um)
    
    # initialize iteration counter and output vectors
    i = 0
    affs = []
    angles = []
    
    # output 
    if save2disk and folder is not None:        
        filename = '_it_' + str(i)
        plt.imsave(folder + '/' + filename + '.png', u, cmap='gray')
    
    #%% run flow
    it = 0  # couter for proximal backtracking
    it_max = 10 # maximum number of prox evaluations
    while i < i_max:            
        
        # define proximal parameter
        if i == 0:
            sg = L.adjoint(L(um))
                    
        norm_sg = H(sg)
        alpha = dt / norm_sg         
       
        # define data fidelity    
        f = (0.5/alpha) * odl.solvers.L2NormSquared(X).translated(um)
        
        # evaluate the proximal operator    
        odl.solvers.pdhg(u,f,g,L,1000,tau,sigma) 
        
        # compute subgradient
        sg = (um - u)/alpha                         
        
        # subtract mean to be on the safe side
        u -= proj(u)

        # normalize to create the iterate
        u /= 1-dt/H(um)     
        
        # evaluate Rayleigh quotient
        ev = J(u)/H(u-proj(u))
    
        # check whether Rayleigh quotient decreased
        if ev < ev_m:
            it = 0            
            
            # compute eigenvector affinity in <= 1
            aff = odl.solvers.L2NormSquared(X)(sg)/J(sg)
            affs.append(aff)
            
            # evaluate angle
            # angle = 1 - um.inner(u)/(H(u)*H(um))
            angle = 1 - sg.inner(u)/(H(sg)*H(u))
            angles.append(angle)
                
            # output
            print(20*'<>')                      
            print('Affinity: {:.2f}'.format(aff))
            print('Angle: {:.2E} '.format(angle))
            print('Subdiff Eigenvalue: {:.2f} '.format(ev))
            print(20*'<>')                      
        
            # update iteration counter and plot image
            i += 1
            u.show('Iteration ' + str(i))
            
            # save result to disk
            if save2disk and folder is not None:        
                filename = 'it_' + str(i)
                plt.imsave(folder + '/' + filename + '.png', u, cmap='gray')
                
            # update old variables    
            um = u.copy()
            ev_m = ev
        else:
            it += 1
            
            if it >= it_max:
                
                #evaluate angle                
                tmp = um.copy()
                f = (0.5/alpha) * odl.solvers.L2NormSquared(X).translated(tmp)        
                odl.solvers.pdhg(um,f,g,L,1000,tau,sigma) 
                angle = 1-um.inner(tmp)/(H(um)*H(tmp))
                angles.append(angle)
                
                #evaluate affinity
                sg = (um-tmp)/alpha
                aff = odl.solvers.L2NormSquared(X)(sg)/J(sg)
                affs.append(aff)
                
                print('Stopped iterations because Rayleigh quotient cannot be improved')
                break
            else:
                continue        
     
    return um, affs, angles     
##############################################################################        
