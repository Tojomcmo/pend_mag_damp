import numpy as np


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
w_n = np.sqrt(g/L)
b = b * num_mags
m = m * num_mags

z = np.linspace(0.2, 1, 9)

l_simple     = [None] * len(z)
l_m_included = [None] * len(z)

for idx in range(len(z)):
    l_simple[idx] = np.sqrt((2 * M * z[idx])*(1/b) * np.sqrt(g * L**3))
    l_test = 0
    z_test = 0
    while z_test < z[idx]:
        l_test += 0.001
        z_test = (b * l_test**2) / ( 2 * np.sqrt(g * (m*l_test + M*L) * (m*l_test**2 + M*L**2)))
    l_m_included[idx] = l_test
print('l_simple: ' ,l_simple)    
print('l_with mag mass: ' ,l_m_included)    