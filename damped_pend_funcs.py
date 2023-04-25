import numpy as np
import time
import jax.numpy as jnp

import misc_funcs as mf



# thddot + (b/m * l^2 / L^2) thdot - g/L sin th
# damping ratio ranges: (0.1 - 1.3)
# K = mgL
# w_n = sqrt(K/I) = sqrt(mgL/mL^2) = sqrt(g/L)


# natural frequency = sqrt(k/m)
# damping ratio = 2*m*sqrt(k/m) = 2*m*wn


# I = m*L^2
# B = z * sqrt(w_n * I)
# K = I * w_n^2


#   mass spring damper TF: H(s) = 1 / (m*s^2 + b*s + k)
#   w_n = sqrt(k/m) 
#   z   = b / (2*sqrt(km))
#
#   define m as 1, w_n as 1, z as input
#   m = 1
#   k = m * w_n^2 = 1
#   b = 2 * z sqrt(w_n * m) = 2 * z

# two point masses:
#    (ml^2 + ML^2)thddot + (bl^2)thdot + (g(ml + ML))sin(th)
# linearize:
#    (ml^2 + ML^2)thddot + (bl^2)thdot + (g(ml + ML))th
# calc natural frequency:
#     w_n = sqrt(k/m) = sqrt((g(ml + ML)) / (ml^2 + ML^2))
# calc damping ratio:
#     z = b / (2*sqrt(k*m)) = bl^2 / (2 * sqrt( (g(ml + ML)) * (ml^2 + ML^2)))

      

g = 9.81
L = 0.25
M = 0.05
m = 0.00423
b = 0.3
num_mags = 2

I = M*L**2
K = M*g*L
w_n = jnp.sqrt(g/L)
b = b * num_mags
m = m * num_mags
z = jnp.linspace(0.2, 1, 100)

##### No mag mass #######
l_massless = [None] * len(z)
tstart     = time.time()

for idx in range(len(z)):
    l_massless[idx] = jnp.sqrt((2 * M * z[idx])*(1/b) * jnp.sqrt(g * L**3))

time_elapsed = time.time() - tstart    
for i in l_massless:
    print(i)   
print('time elapsed simple search:  ', time_elapsed)    
##### simple search #######

l_m_included = [None] * len(z)
tstart = time.time()

for idx in range(len(z)):
    if idx == 0:
        l_test = 0
    z_test = 0
    while z_test < z[idx]:
        l_test += 0.001
        z_test = (b * l_test**2) / ( 2 * jnp.sqrt(g * (m*l_test + M*L) * (m*l_test**2 + M*L**2)))
    l_m_included[idx] = l_test
time_elapsed = time.time() - tstart      
print('l_with mag mass: ')
for i in l_m_included:
    print(i)   
print('time elapsed simple search:  ', time_elapsed)


###### Newton Raphson solution ########

l_sol = [None] * len(z)
eps   = 1e-6
tstart = time.time()

for idx in range(len(z)):
    #set initial guess and test value
    if idx == 0:
        l_test = 0.1    
    #create function of z zeroed at target z value    
    z_wrt_l_func = lambda l: (mf.calculate_pend_z_with_mag_mass(l, L, M, m, g, b) - z[idx])   
    #create initial test variable 
    z_test = z_wrt_l_func(l_test)
    #while test variable is out of tolerance, run NR method
    while jnp.abs(z_test) > eps:
        l_test = mf.NRmethod(z_wrt_l_func, l_test) 
        z_test = z_wrt_l_func(l_test)
    l_sol[idx] = l_test

time_elapsed = time.time() - tstart   

print('l from newton raphson')
for i in l_sol:
    print(i)   

print('time elapsed NR:  ', time_elapsed)
