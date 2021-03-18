import odl 
import numpy as np

X = odl.uniform_discr([-1, -1], [1, 1], [100, 100], interp='linear')

iso = False
p_grad = 1
sig = 0.1*0
testcase = 'disk'
stepsize = 0.01

if testcase == 'disk':
    data_func = lambda x: x[0]**2 + x[1]**2 < 0.7**2    
elif testcase == 'square': 
    data_func = odl.phantom.cuboid(X)
elif testcase == 'noise':
    data_func = odl.phantom.white_noise(X)
    
data = X.element(data_func)
# data = odl.phantom.cuboid(X) -odl.phantom.geometric.defrise(X, nellipses=1)
# data = odl.phantom.tgv_phantom(X)
# data = odl.phantom.white_noise(X)

noise = odl.phantom.white_noise(X, mean=0, stddev = sig)
data += noise

    
###########
H = odl.solvers.L2Norm(X)
proj = np.mean
            
# data -= proj(data)
data /= H(data)

if p_grad == 1:
    if iso is True:
        g = odl.solvers.GroupL1Norm(X.tangent_bundle, 2)
    else:
        g = odl.solvers.GroupL1Norm(X.tangent_bundle, 1)
elif p_grad == 2:    
    g = odl.solvers.L2Norm(X.tangent_bundle)

L = odl.Gradient(X, method='forward', pad_mode='symmetric')
opNorm = 2*np.sqrt(sum(1/X.cell_sides**2))
L.norm = opNorm
J = g * L

(tau, sigma) = odl.solvers.pdhg_stepsize(L.norm)

#%%
RQm = np.inf
u = data.copy()
um = u.copy()

c = 0.95

u.show('Iteration ' + str(0))


for i in range(1,100):
    

    f = (0.5/stepsize) * odl.solvers.L2NormSquared(X).translated(um)
    
    
    odl.solvers.pdhg(u,f,g,L,1000,tau,sigma) 
    
    sg = (um - u)/stepsize
    
    affinity = H(sg)**2/J(sg)
           
    RQ = J(u)/H(u)

    if RQ <= RQm:
        print('------------------------------------------')
        print('Eigenvalue '+ str(RQ))
        sg.show('Iteration ' + str(i))
     
        um = u.copy()
        RQm = RQ
    else:
        continue
   
    
       
        


        
